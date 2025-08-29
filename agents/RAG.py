import os
os.environ['HF_HOME'] = '/home/tanxh/.cache/huggingface'


import json
import yaml
import torch
from pathlib import Path
from openai import OpenAI
from pymilvus import connections, Collection
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 公用参数，可适配不同知识库的检索
config_path = 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def get_local_model_path(model_name: str) -> str:
    """
    将 'Alibaba-NLP/gte-multilingual-base' 转换成本地 cache 路径
    如 ~/.cache/huggingface/hub/models--Alibaba-NLP--gte-multilingual-base/snapshots/<hash>
    """
    cache_dir = '/home/tanxh/.cache/huggingface/hub'
    cache_dir = Path(cache_dir).resolve()

    safe_name = model_name.replace("/", "--")
    model_dir = cache_dir / f"models--{safe_name}" / "snapshots"
    
    # 默认只取第一个 snapshot 子目录（一般只有一个）
    snapshot_dirs = list(model_dir.glob("*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for model: {model_name}")
    
    return str(snapshot_dirs[0])


rag_conf = config['rag']
model_name_embed = rag_conf['embedding_model']
model_name_rank = rag_conf['ranking_model']


# 获取本地路径
# embedding_model_path = get_local_model_path(model_name_embed)
# print(embedding_model_path)
ranking_model_path = get_local_model_path(model_name_rank)
# print(ranking_model_path)


device = rag_conf.get("device", "cpu")
# embedding_model = SentenceTransformer(
#     embedding_model_path,
#     device = device,
#     trust_remote_code = True,
#     local_files_only = True,
# )


ranking_tokenizer = AutoTokenizer.from_pretrained(ranking_model_path, local_files_only=True)
ranking_model = AutoModelForSequenceClassification.from_pretrained(
    ranking_model_path,
    trust_remote_code = True,
    local_files_only = True,
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
)
ranking_model.eval()


def emb_text(text: list[str]):
    """
    文本向量化
    """
    # return embedding_model.encode(text, normalize_embeddings=True).tolist()
    client = OpenAI(
        base_url = rag_conf['base_url'],
        api_key = rag_conf['api_key'],
    )

    completion = client.embeddings.create(
        model = model_name_embed,
        input = text,
        encoding_format = "float"
    )

    content = completion.model_dump_json()
    content_dict = json.loads(content)

    # rep = content_dict['data'][0]['embedding']
    # return [rep]
    return [item['embedding'] for item in content_dict['data']]


def rerank(query: str, docs: list[str], rerank_topk: int) -> list[dict]:
    """
    使用 ranking 模型对检索到的文档进行重排
    """
    pairs = [[query, doc] for doc in docs]
    inputs = ranking_tokenizer(
        pairs,
        padding = True,
        truncation = True,
        return_tensors = 'pt',
        max_length = 512
    )

    with torch.no_grad():
        scores = ranking_model(**inputs, return_dict=True).logits.view(-1).float()

    scored_docs = list(zip(docs, scores.tolist()))
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [
        {"text": text, "score": score} for text, score in ranked_docs[:rerank_topk]
    ]


class ContentRetriever():
    """
    根据 query 检索摘要获取相关内容的类
    """
    def __init__(self):
        # Milvus 配置
        milvus_conf = config['milvus']
        connections.connect(uri=milvus_conf['uri'], token=milvus_conf['token'], db_name=milvus_conf['db'])

        collection_conf = config['collections']
        self.summary_collection = Collection(collection_conf['summary'])
        self.content_collection = Collection(collection_conf['content'])
        self.s2c_collection = Collection(collection_conf['summary2content'])

        # 知识库检索的参数
        self.search_topk_metric = rag_conf['search_topk_metric']
        self.rerank_topk_metric = rag_conf['rerank_topk_metric']
        self.search_topk_other = rag_conf['search_topk_other']
        self.rerank_topk_other = rag_conf['rerank_topk_other']
        self.content_topk = rag_conf['content_topk']


    def search_content_from_summary(self, query_text: str, source: list = None) -> dict:
        """
        先匹配摘要，再从摘要中获取具体的内容
        """
        embeddings = emb_text([query_text])
        # print(type(embeddings))

        search_topk = self.search_topk_other
        rerank_topk = self.rerank_topk_other

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        # 1. 对 query 进行向量检索，检索【摘要】集合，返回相似度更高的 topk1 个摘要与其 id
        # source = ['system'、'lustre'、'promql'、'slurm'、'handbook'、'ticket'、'general']
        search_res = self.summary_collection.search(
            data = embeddings,
            anns_field = "vector",
            param = search_params,
            expr = f'source in {source}' if source else None,
            limit = search_topk,
            output_fields = ['id', 'text'],
        )

        summary_ids = [hit.entity.get("id") for hit in search_res[0]]
        summaries = [hit.entity.get("text") for hit in search_res[0]]
        summary_id_map = {text: id for text, id in zip(summaries, summary_ids)}

        # 2. 对摘要进行重排，返回 topk2 个摘要与其 id
        reranked_summaries = rerank(query_text, summaries, rerank_topk)
        summary_texts = [item["text"] for item in reranked_summaries]
        reranked_summary_ids = [summary_id_map[item["text"]] for item in reranked_summaries]

        # 3. 检索摘要与内容关联的集合，根据摘要获取其相应内容
        s2c = self.s2c_collection.query(expr=f"s_id in {reranked_summary_ids}", output_fields=["c_id"])
        content_ids = [item["c_id"] for item in s2c]

        # 4. 返回摘要下的全部内容
        contents = self.content_collection.query(expr=f"id in {content_ids}", output_fields=["text"])
        content_texts = [item["text"] for item in contents]

        return summary_texts, content_texts


    def search_only_content(self, query_text: str) -> dict:
        """
        仅检索内容集合
        """
        embeddings = emb_text([query_text])

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        topk = self.content_topk

        # 1. 对 query 进行向量检索，检索【内容】集合，返回相似度更高的 topk1 个内容与其 id
        search_res = self.content_collection.search(
            data = embeddings,
            anns_field = "vector",
            param = search_params,
            limit = topk,
            output_fields = ['text'],
        )

        contents = [hit.entity.get("text") for hit in search_res[0]]

        # 2. 对内容进行重排，返回 topk2 个内容与其 id
        reranked_contents = rerank(query_text, contents, topk)

        return [item["text"] for item in reranked_contents]


    def get_rag_from_summary2content(self, query: str, source: list) -> list[Document]:
        """
        input：用户询问，期望知识来源
        output：检索出的相关文档以及原始数据的字典

        接口调用：
        from RAG import ContentRetriever

        def get_rag(query: str, source: str):
            retriever = ContentRetriever()

            docs = retriever.get_rag_from_summary2content(query, source)
            context = "\n\n".join(doc.page_content for doc in docs)

            return context

        def main():
            query = ''
            source = ''
            rag_context = get_rag(query, source)
        """
        _, contents = self.search_content_from_summary(query, source)
        docs = [Document(page_content=ctx) for ctx in contents]
        return docs


    def get_rag_only_content(self, query: str) -> list[Document]:
        contents = self.search_only_content(query)
        docs = [Document(page_content=ctx) for ctx in contents]
        return docs


def main() -> None:
    retriever = ContentRetriever()
    query = '如何让 yhacct 输出作业的最⼤虚拟内存、最⼤ RSS 和平均 RSS 等详细信息？'
    source = ['slurm']
    res = retriever.get_rag_from_summary2content(query, source)
    # res = retriever.get_rag_only_content(query)
    print(res)


if __name__ == '__main__':
    main()


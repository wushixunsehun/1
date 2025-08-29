import os
import time
import yaml
import argparse
import public_function as pf
from MilvusAPI import Milvus
from doc_process import read_docs


dataset_postfix = {
    'html2md': '.md',
    'markdown': '.md',
    'pdf': '.pdf',
    'docx': '.docx',
    'excel': '.xlsx'
}


def milvus_run(milvus: Milvus, embedding_model, summary_generator):
    """处理数据集并插入知识库"""
    docs = {}
    dataset = args.dataset
    print(f'dataset: {dataset}')

    dataset_dir = os.path.join('dataset', dataset)
    preprocess_data_dir = os.path.join('preprocess_data_files', dataset)


    summary_path = os.path.join(preprocess_data_dir, 'summary_500/')
    os.makedirs(summary_path, exist_ok=True)
    files = [os.path.join(summary_path, file) for file in os.listdir(summary_path) if file.endswith('.txt')]

    if not files:
        docs = read_docs(dataset_dir, dataset, embedding_model, summary_generator)
    else:
        for file in files:
            try:
                doc_name = os.path.basename(file).replace('.txt', dataset_postfix[dataset])
                raw_file_path = os.path.join(dataset_dir, doc_name)
                doc_abs_path = os.path.abspath(raw_file_path)

                docs[doc_abs_path] = []

                data = pf.read_data_from_txt(file)
                docs[doc_abs_path].extend(data)
            except Exception as e:
                print(f"读取文件{file}时发生错误：{e}")

    # docs = read_docs(dataset_dir, dataset, embedding_model, summary_generator)

    # 空值检查，避免知识库污染
    for _, doc in docs.items():
        for docs_data in doc:
            summary_text = docs_data.get('summary', None)
            chunks = docs_data.get('chunks', None)

            if not summary_text:
                raise ValueError("文档摘要存在空值")

            if not chunks:
                raise ValueError("文档内容存在空值")

    # 文档嵌入
    summary_data, content_data, s2c_data = milvus.docs_embedding(docs)

    # 插入数据
    milvus.insert_data(summary_data, content_data, s2c_data)


def main() -> None:
    config_path = 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    milvus_config = config['milvus']
    embedding_model = pf.load_embedding_model(args.emb_model, milvus_config['device'])
    summary_generator = pf.load_summary_model(args.sum_model, config['llm'])
    milvus_config['embedding_model'] = embedding_model

    milvus = Milvus(milvus_config)
    # milvus.list_collections()

    milvus_run(milvus, embedding_model, summary_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='excel')
    parser.add_argument('--emb_model', default='Alibaba-NLP/gte-multilingual-base')
    parser.add_argument('--sum_model', default='qwq_local') # [qwq_local, qwq_api, t5]
    args = parser.parse_args()

    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f'Run time: {(end - start):.4f}s')

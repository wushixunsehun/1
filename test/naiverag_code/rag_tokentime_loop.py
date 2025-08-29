import tiktoken
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

# 导入所需库
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

# 全局变量，只初始化一次
embeddings = None
llm = None
vector_store = None

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    使用tiktoken计算文本的token数量
    
    Args:
        text: 待计算的文本内容
        model: 目标模型名称，用于选择对应编码
    
    Returns:
        文本的token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def load_and_split_documents(
    file_path: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    加载文档并进行文本分割
    
    Args:
        file_path: 文档文件路径
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
    
    Returns:
        分割后的文档列表
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def create_vector_store(documents: List[Document]) -> FAISS:
    """
    创建向量存储
    
    Args:
        documents: 文档列表
    
    Returns:
        FAISS向量存储实例
    """
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
    return FAISS.from_documents(documents, embeddings)


def build_llm_chain() -> tuple:
    """
    构建LLM链和检索器
    
    Returns:
        (检索链, 提示词模板) 元组
    """
    global llm
    if llm is None:
        llm = ChatOpenAI(
            model="Qwen/Qwen3-30B-A3B",
            base_url="http://a6000-G5500-V6:5414/v1",
            api_key="EMPTY"
        )
    
    # 定义提示词模板
    prompt = ChatPromptTemplate.from_template("""
    仅根据提供的内容回答问题，要求回答是一段文字(不限制字数)，且不能写给出文字之外的知识，
    鼓励列点回答(标注1.2.3.)，文字不能加粗，列点前要求有综述性的文字和冒号，不要换行或者分段，
    回答简洁:
    <context>
    {context}
    </context>
    Question: {input}
    """)
    
    # 构建文档处理链
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 构建检索器
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain, prompt


def process_question(
    retrieval_chain: object, 
    prompt: ChatPromptTemplate, 
    documents: List[Document], 
    question: str
) -> Dict[str, str]:
    """
    处理用户问题并获取回答
    
    Args:
        retrieval_chain: 检索链实例
        prompt: 提示词模板
        documents: 文档列表
        question: 用户问题
    
    Returns:
        包含回答的字典
    """
    start_time = time.time()
    
    # 执行检索链
    response = retrieval_chain.invoke({"input": question})
    
    end_time = time.time()
    time_cost = end_time - start_time
    print(f"模型响应时间: {time_cost:.2f} 秒")
    
    # 计算token数量
    prompt_text = prompt.format_messages(
        context="\n".join([doc.page_content for doc in documents]),
        input=question
    )
    prompt_tokens = sum(count_tokens(str(msg)) for msg in prompt_text)
    completion_tokens = count_tokens(response["answer"])
    total_tokens = prompt_tokens + completion_tokens

    print(f"总token数: {total_tokens}")
    print(f"提示词token数: {prompt_tokens}")
    print(f"生成内容token数: {completion_tokens}")
    
    return response, time_cost, total_tokens, prompt_tokens, completion_tokens


def save_answer(answer: str, file_path: str, time_cost: float, total_tokens: int, 
                prompt_tokens: int, completion_tokens: int) -> None:
    """
    保存回答到文件
    
    Args:
        answer: 回答内容
        file_path: 保存路径
        time_cost: 响应时间
        total_tokens: 总token数
        prompt_tokens: 提示词token数
        completion_tokens: 生成内容token数
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            answer_o = answer.replace('\n', '')
            print(answer_o)
            file.write(answer_o)
            file.write('\n')
            file.write(f"响应时间:{time_cost:.2f} 秒\n")
            file.write(f"总token数: {total_tokens}\n")
            file.write(f"提示词token数: {prompt_tokens}\n")
            file.write(f"生成内容token数: {completion_tokens}\n")
        print(f"内容已成功写入文件: {file_path}")
    except Exception as e:
        print(f"写入文件时出错: {str(e)}")


def clear_gpu_memory():
    """清理GPU内存"""
    import gc
    import torch
    
    # 删除不再需要的对象
    global vector_store
    vector_store = None
    
    # 强制垃圾回收
    gc.collect()
    
    # 释放PyTorch缓存内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_file(input_file_idx: int, output_file_idx: int) -> None:
    """
    处理单个文件的函数
    
    Args:
        input_file_idx: 输入文件编号
        output_file_idx: 输出文件编号
    """
    global vector_store
    
    # 配置文件路径
    document_path = f"./file_response5/response{input_file_idx}.txt"
    answer_path = f"./file_answer_naiverag5/answer{output_file_idx}.txt"
    
    # 检查输入文件是否存在
    if not Path(document_path).exists():
        print(f"警告: 输入文件不存在: {document_path}")
        return
    
    # 读取查询内容
    try:
        with open(document_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            query = input_data.get("query", "未找到查询内容")
            print(f"处理文件 {input_file_idx}: {query}")
    except json.JSONDecodeError:
        print(f"错误: 文件 {document_path} 不是有效的JSON格式")
        return
    
    # 1. 加载并分割文档
    documents = load_and_split_documents(document_path)
    
    # 2. 创建向量存储
    vector_store = create_vector_store(documents)
    
    # 3. 构建LLM链
    retrieval_chain, prompt = build_llm_chain()
    
    # 4. 处理问题并获取回答
    response, time_cost, total_tokens, prompt_tokens, completion_tokens = process_question(
        retrieval_chain=retrieval_chain,
        prompt=prompt,
        documents=documents,
        question=query
    )

    # 5. 保存回答
    save_answer(
        response["answer"], 
        answer_path, 
        time_cost, 
        total_tokens, 
        prompt_tokens, 
        completion_tokens
    )
    print(f"文件 {input_file_idx} 处理完成，结果保存在 {answer_path}\n")
    
    # 清理当前文件处理占用的GPU内存
    clear_gpu_memory()


def main(start_position: int = 1, loop_count: int = 3) -> None:
    """
    主函数：执行循环处理多个文件
    
    Args:
        start_position: 起始位置
        loop_count: 循环次数
    """
    print(f"开始处理文件，起始位置: {start_position}，循环次数: {loop_count}")
    
    for i in range(loop_count):
        input_idx = start_position + i
        output_idx = start_position + i
        
        print(f"\n===== 处理文件 {input_idx}/{output_idx} =====")
        process_file(input_idx, output_idx)
    
    print(f"所有 {loop_count} 个文件处理完成")


if __name__ == "__main__":
    # 示例：从文件1开始，处理3个文件（即response1/2/3.txt -> answer1/2/3.txt）
    main(start_position=201, loop_count=10)
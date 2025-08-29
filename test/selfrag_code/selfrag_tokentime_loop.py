from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from langchain_core.output_parsers import StrOutputParser
import time
from langchain.schema import AIMessage
import logging
import tiktoken
import json
from typing import Callable, Tuple, List, Dict
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

# 全局变量：LLM和Embedding
llm = None
embeddings = None

def setup_logging() -> logging.Logger:
    """配置日志系统，返回日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """计算文本的token数量，包含异常处理"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def calculate_prompt_tokens(prompt: ChatPromptTemplate, inputs: dict, model: str) -> int:
    """计算提示词的总token数"""
    formatted_prompt = prompt.format_messages(**inputs)
    total_tokens = 0
    for message in formatted_prompt:
        message_text = f"{message.type}: {message.content}"
        total_tokens += count_tokens(message_text, model)
    return total_tokens

def setup_retriever(file_path: str, logger: logging.Logger) -> FAISS:
    """设置检索器，使用全局embeddings"""
    global embeddings
    try:
        start_time = time.time()
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(doc_splits, embeddings)
        logger.info(f"检索器初始化耗时: {time.time() - start_time:.2f} 秒")
        return vectorstore
    except Exception as e:
        logger.error(f"检索器初始化失败: {str(e)}")
        raise

def setup_global_llm_and_embeddings(logger: logging.Logger) -> None:
    """初始化全局LLM和Embedding（仅执行一次）"""
    global llm, embeddings
    if llm is not None and embeddings is not None:
        return  # 已初始化则跳过
    
    # 初始化LLM
    try:
        start_time = time.time()
        llm = ChatOpenAI(
            model="Qwen/Qwen3-30B-A3B",
            base_url="http://a6000-G5500-V6:5414/v1",
            api_key="EMPTY"
        )
        logger.info(f"LLM初始化耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        logger.error(f"LLM初始化失败: {str(e)}")
        raise
    
    # 初始化Embedding
    try:
        start_time = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Embedding初始化耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        logger.error(f"Embedding初始化失败: {str(e)}")
        raise

def setup_retrieval_grader() -> Tuple[Callable, Callable[[dict, str], int]]:
    """设置检索评分器，评估文档与问题的相关性"""
    global llm
    system = """您是一名评估检索到的文档与用户问题相关性的评分员。\n 
        它不需要是一个严格的测试。目标是过滤掉错误的检索。\n
        如果文档包含与用户问题相关的关键字或语义，请将其评为相关。\n
        给出"是"或"否"，回答的内容不带标点符号。以表明文档是否与问题相关。"""
    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")]
    )
    
    def calculate_tokens(inputs: dict, model: str) -> int:
        return calculate_prompt_tokens(grade_prompt, inputs, model)
    
    return grade_prompt | llm, calculate_tokens

def modify_rag_prompt(logger: logging.Logger) -> ChatPromptTemplate:
    """修改RAG提示模板"""
    start_time = time.time()
    prompt = ChatPromptTemplate.from_template("""
    仅根据提供的内容用中文回答问题，要求回答是一段文字(不限制字数)，且不能写给出文字之外的知识，
    鼓励列点回答(标注1.2.3.)，文字不能加粗，列点前要求有综述性的文字和冒号，不要换行或者分段，
    回答简洁:
    <context>
    {context}
    </context>
    Question: {input}
    """)  
    logger.info(f"提示词修改耗时: {time.time() - start_time:.2f} 秒")
    return prompt

def setup_rag_chain(prompt: ChatPromptTemplate, vector_store: FAISS) -> Tuple[Callable, Callable[[dict, str], int]]:
    """设置检索链，补充上下文token计算逻辑"""
    global llm
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    def calculate_tokens(inputs: dict, model: str, documents: List[Document]) -> int:
        """计算包含上下文和问题的总token数"""
        context = "\n".join([doc.page_content for doc in documents])
        formatted_prompt = prompt.format_messages(context=context, input=inputs["input"])
        total_tokens = sum(count_tokens(str(msg), model) for msg in formatted_prompt)
        return total_tokens
    
    return retrieval_chain, calculate_tokens

def setup_hallucination_grader() -> Tuple[Callable, Callable[[dict, str], int]]:
    """设置幻觉评分器，评估答案是否基于事实"""
    global llm
    system = """您是一名评分员，职责是评估LLM生成的答案是否基于/支持一组检索到的事实。\n 
                请您给出"是"或"否"。回答的内容不带标点符号。"是"意味着答案基于/支持一系列事实。"否"意味着答案不基于/不支持一系列事实。"""
    
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")]
    )
    
    def calculate_tokens(inputs: dict, model: str) -> int:
        return calculate_prompt_tokens(hallucination_prompt, inputs, model)
    
    return hallucination_prompt | llm, calculate_tokens

def setup_answer_grader() -> Tuple[Callable, Callable[[dict, str], int]]:
    """设置答案评分器，评估答案是否解决问题"""
    global llm
    system = """您是一名评分员，职责是评估答案是否解决问题。\n 
               给出"是"或"否"。回答的内容不带标点符号。"是"意味着答案解决了问题。"否"意味着答案没有解决问题。"""
    
    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")]
    )
    
    def calculate_tokens(inputs: dict, model: str) -> int:
        return calculate_prompt_tokens(answer_prompt, inputs, model)
    
    return answer_prompt | llm, calculate_tokens

def setup_question_rewriter() -> Tuple[Callable, Callable[[dict, str], int]]:
    """设置问题重写器，用于优化检索问题"""
    global llm
    system = """您是一个重写器，职责是将输入问题转换为优化的更好版本的问题。\n 
                要求用中文重写问题。\n
                重写的新问题用于矢量库检索。查看输入并尝试推理潜在的语义意图/含义."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")]
    )
    
    def calculate_tokens(inputs: dict, model: str) -> int:
        return calculate_prompt_tokens(re_write_prompt, inputs, model)
    
    return re_write_prompt | llm | StrOutputParser(), calculate_tokens

def save_answer_with_metrics(answer: str, token_data: dict, time_data: dict, file_path: str) -> None:
    """保存答案及token、时间数据到文件"""
    try:
        data = {
            "answer": answer,
            "token_data": token_data,
            "time_data": time_data
        }
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"答案、token及时间数据已写入文件: {file_path}")
    except Exception as e:
        print(f"写入文件时出错: {str(e)}")

def process_file(file_num: int, logger: logging.Logger) -> None:
    """处理单个文件"""
    file_path = f"./file_response3/response{file_num}.txt"
    answer_path = f"./file_answer_selfrag3/answer{file_num}.txt"
    last_answer_path = f"./file_answer_selfrag3/answer{file_num}.txt"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            query = input_data.get("query", "未找到查询内容")
            print(f"\n--- 处理文件 {file_path} ---")
            print(f"查询问题: {query}")
        
        vector_store = setup_retriever(file_path, logger)
        
        # 获取组件及其token计算函数
        retrieval_grader, calc_retrieval_tokens = setup_retrieval_grader()
        modified_prompt = modify_rag_prompt(logger)
        rag_chain, calc_rag_tokens = setup_rag_chain(modified_prompt, vector_store)
        hallucination_grader, calc_halluc_tokens = setup_hallucination_grader()
        answer_grader, calc_answer_tokens = setup_answer_grader()
        question_rewriter, calc_rewrite_tokens = setup_question_rewriter()
        
        max_attempts = 3
        current_attempt = 1
        current_question = query
        last_generation = ""
        last_token_data = None
        last_time_data = None

        while current_attempt <= max_attempts:
            print(f"\n=== 尝试 {current_attempt}/{max_attempts} ===")
            print(f"问题: {current_question}")
            
            attempt_tokens = 0
            attempt_time = 0
            current_token_data = {
                "retrieval_evaluation": 0,
                "rag_prompt": 0,
                "generation": 0,
                "hallucination_evaluation": 0,
                "answer_evaluation": 0,
                "rewrite": 0,
                "total": 0
            }
            current_time_data = {
                "retrieval_evaluation": 0,
                "rag_prompt": 0,
                "generation": 0,
                "hallucination_evaluation": 0,
                "answer_evaluation": 0,
                "rewrite": 0,
                "total": 0
            }
            
            # 获取检索文档
            docs = vector_store.similarity_search(current_question)
            doc_txt = docs[1].page_content if len(docs) > 1 else ""
            
            # 检索评估阶段
            retrieval_start_time = time.time()
            retrieval_inputs = {"question": current_question, "document": doc_txt}
            prompt_tokens = calc_retrieval_tokens(retrieval_inputs, "gpt-3.5-turbo")
            retrieval_score = retrieval_grader.invoke(retrieval_inputs)
            answer_tokens = count_tokens(
                retrieval_score.content if isinstance(retrieval_score, AIMessage) else str(retrieval_score),
                "gpt-3.5-turbo"
            )
            stage_tokens = prompt_tokens + answer_tokens
            current_token_data["retrieval_evaluation"] = stage_tokens
            attempt_tokens += stage_tokens
            retrieval_duration = time.time() - retrieval_start_time
            current_time_data["retrieval_evaluation"] = retrieval_duration
            attempt_time += retrieval_duration
            print(f"检索评估耗时: {retrieval_duration:.2f} 秒，token消耗: {stage_tokens}")
            print(f"检索相关性评分: {retrieval_score.content.strip()}")
            
            # RAG提示词阶段（时间已包含在生成阶段，此处仅记录token）
            rag_inputs = {"input": current_question}
            rag_prompt_tokens = calc_rag_tokens(rag_inputs, "gpt-3.5-turbo", docs)
            current_token_data["rag_prompt"] = rag_prompt_tokens
            attempt_tokens += rag_prompt_tokens
            print(f"RAG提示词token: {rag_prompt_tokens}")
            
            # 生成回答阶段
            generation_start_time = time.time()
            response = rag_chain.invoke({"input": current_question})
            generation = response["answer"]
            generation_duration = time.time() - generation_start_time
            current_time_data["generation"] = generation_duration
            attempt_time += generation_duration
            print(f"回答生成耗时: {generation_duration:.2f} 秒")
            
            generation_tokens = count_tokens(generation, "gpt-3.5-turbo")
            current_token_data["generation"] = generation_tokens
            attempt_tokens += generation_tokens
            print(f"回答生成token消耗: {generation_tokens}")
            
            # 幻觉评估阶段
            hallucination_start_time = time.time()
            hallucination_inputs = {"documents": [doc.page_content for doc in docs], "generation": generation}
            prompt_tokens = calc_halluc_tokens(hallucination_inputs, "gpt-3.5-turbo")
            hallucination_score = hallucination_grader.invoke(hallucination_inputs)
            answer_tokens = count_tokens(
                hallucination_score.content if isinstance(hallucination_score, AIMessage) else str(hallucination_score),
                "gpt-3.5-turbo"
            )
            stage_tokens = prompt_tokens + answer_tokens
            current_token_data["hallucination_evaluation"] = stage_tokens
            attempt_tokens += stage_tokens
            hallucination_duration = time.time() - hallucination_start_time
            current_time_data["hallucination_evaluation"] = hallucination_duration
            attempt_time += hallucination_duration
            print(f"幻觉评估耗时: {hallucination_duration:.2f} 秒，token消耗: {stage_tokens}")
            print(f"幻觉评分: {hallucination_score.content.strip()}")
            
            # 答案评估阶段
            answer_start_time = time.time()
            answer_inputs = {"question": current_question, "generation": generation}
            prompt_tokens = calc_answer_tokens(answer_inputs, "gpt-3.5-turbo")
            answer_score = answer_grader.invoke(answer_inputs)
            answer_tokens = count_tokens(
                answer_score.content if isinstance(answer_score, AIMessage) else str(answer_score),
                "gpt-3.5-turbo"
            )
            stage_tokens = prompt_tokens + answer_tokens
            current_token_data["answer_evaluation"] = stage_tokens
            attempt_tokens += stage_tokens
            answer_duration = time.time() - answer_start_time
            current_time_data["answer_evaluation"] = answer_duration
            attempt_time += answer_duration
            print(f"答案评估耗时: {answer_duration:.2f} 秒，token消耗: {stage_tokens}")
            print(f"答案评分: {answer_score.content.strip()}")
            
            # 问题重写阶段（若需要）
            if current_attempt < max_attempts and not all([
                hallucination_score.content.strip() == "是",
                answer_score.content.strip() == "是"
            ]):
                print("\n答案未通过评估，正在重写问题...")
                rewrite_start_time = time.time()
                rewrite_inputs = {"question": current_question}
                prompt_tokens = calc_rewrite_tokens(rewrite_inputs, "gpt-3.5-turbo")
                current_question = question_rewriter.invoke(rewrite_inputs)
                answer_tokens = count_tokens(current_question, "gpt-3.5-turbo")
                stage_tokens = prompt_tokens + answer_tokens
                current_token_data["rewrite"] = stage_tokens
                attempt_tokens += stage_tokens
                rewrite_duration = time.time() - rewrite_start_time
                current_time_data["rewrite"] = rewrite_duration
                attempt_time += rewrite_duration
                print(f"问题重写耗时: {rewrite_duration:.2f} 秒，token消耗: {stage_tokens}")
                print(f"重写后的问题: {current_question}")
            
            current_token_data["total"] = attempt_tokens
            current_time_data["total"] = attempt_time
            last_generation = generation
            last_token_data = current_token_data.copy()
            last_time_data = current_time_data.copy()
            
            # 检查评估结果
            if all([
                hallucination_score.content.strip() == "是",
                answer_score.content.strip() == "是"
            ]):
                print(f"\n最终答案: {generation}")
                save_answer_with_metrics(generation, current_token_data, current_time_data, answer_path)
                print(f"本次尝试总耗时: {attempt_time:.2f} 秒")
                print(f"本次尝试总token消耗: {attempt_tokens}")
                break  # 跳出当前文件的尝试循环
        
            current_attempt += 1
        
        else:  # 所有尝试都失败
            print(f"\n已达到最大尝试次数，未能为文件 {file_path} 生成满意的答案。")
            print(f"\n最后一次生成的答案: {last_generation}")
            if last_token_data and last_time_data:
                print(f"最后一次尝试总token消耗: {last_token_data['total']}")
                print(f"最后一次尝试总耗时: {last_time_data['total']:.2f} 秒")
                save_answer_with_metrics(last_generation, last_token_data, last_time_data, last_answer_path)
            else:
                save_answer_with_metrics(last_generation, {"total": 0}, {"total": 0}, last_answer_path)
                
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")

def main():
    """主函数：循环处理多个文件"""
    logger = setup_logging()
    
    # 初始化全局LLM和Embedding（仅一次）
    setup_global_llm_and_embeddings(logger)
    
    # 定义初始文件位置和循环次数
    initial_file_num = 166  # 初始文件为response1.txt
    loop_count = 1        # 循环处理3个文件：response1/2/3.txt
    
    print(f"开始循环处理文件，初始位置: {initial_file_num}, 循环次数: {loop_count}")
    
    for i in range(loop_count):
        # 计算当前文件编号：initial_file_num + i
        file_num = initial_file_num + i
        process_file(file_num, logger)
    
    print("所有文件处理完成")

if __name__ == "__main__":
    main()
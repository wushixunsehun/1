import os
import asyncio
import logging
import logging.config
import json
from pathlib import Path
import shutil
from time import time
import tiktoken
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug

WORKING_DIR = "./dickens"
DATA_BACKUP_DIR = "./data"  # 用于保存dickens目录的备份

def configure_logging():
    """配置日志"""
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo 日志文件: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


# 确保工作目录和备份目录存在
for dir_path in [WORKING_DIR, DATA_BACKUP_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """计算文本的token数量，包含异常处理"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=openai_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def process_iteration(iteration):
    """处理单次循环迭代，从指定编号的response文件读取内容"""
    print(f"\n===== 开始第 {iteration} 次循环 =====")
    
    # 定义本次循环的输入/输出文件路径（从response11开始）
    response_num = 191 + iteration - 1  
    input_dir = f"./file_response4"
    answer_dir = f"./file_answer_lightrag4"
    input_path = os.path.join(input_dir, f"response{response_num}.txt")
    answer_path = os.path.join(answer_dir, f"answer{response_num}.txt")

    total_start_time = time()
    graph_build_time = 0
    query_time = 0
    response = None
    graph_tokens = 0  # 图谱构建阶段的token数
    query_prompt_tokens = 0
    query_completion_tokens = 0

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY 未设置，请设置环境变量后运行程序。"
        )
        print("设置方式: export OPENAI_API_KEY='你的API密钥'")
        return False  # 标记迭代失败

    try:
        #清空dickens目录（保留目录结构，删除文件）
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"删除旧文件: {file_path}")

        rag = await initialize_rag()

        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("测试嵌入函数")
        print(f"测试文本: {test_text}")
        print(f"嵌入维度: {embedding_dim}\n\n")

        graph_start_time = time()
        # 确保输入目录存在
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(input_path):
            print(f"错误: 输入文件 {input_path} 不存在")
            return False

        # 读取JSON格式的输入文件（假设文件结构包含query/summaries/contents）
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            document_text = "\n".join(input_data.get("summaries", [])) + "\n" + "\n".join(input_data.get("contents", []))
            query = input_data.get("query", "未找到查询内容")
            
            # 计算构建图谱的token数（文档内容）
            graph_tokens = count_tokens(document_text)
            print(f"构建图谱的token数量: {graph_tokens}")
            print(f"本次查询的问题: {query}")
            await rag.ainsert(document_text)
        graph_build_time = time() - graph_start_time
        print(f"生成图谱耗时: {graph_build_time:.2f} 秒")
        
        query_start_time = time()
        # 构建提示词（保留原逻辑，从文件中读取query）
        prompt = "仅根据提供的内容用中文回答问题，要求回答是一段文字(不限制字数)，且不能写给出文字之外的知识，\n"
        prompt += "鼓励列点回答(标注1.2.3.)，文字不能加粗，列点前要求有综述性的文字和冒号，不要换行或者分段，\n"
        prompt += str(query)
        query_prompt_tokens = count_tokens(prompt)
        print(f"查询prompt token数: {query_prompt_tokens}")

        response = await rag.aquery(
            prompt,
            param=QueryParam(mode="naive"),
        )
        
        if response:
            completion_text = str(response)
            query_completion_tokens = count_tokens(completion_text)
            print(f"响应内容token数: {query_completion_tokens}")
        query_time = time() - query_start_time
        print(f"查询耗时: {query_time:.2f} 秒")
        print(response)

    except json.JSONDecodeError as e:
        print(f"迭代 {iteration} 解析JSON文件错误: {e}")
        return False
    except Exception as e:
        print(f"迭代 {iteration} 错误: {e}")
        return False
    finally:
        if 'rag' in locals():
            await rag.finalize_storages()

    total_time = time() - total_start_time
    print(f"\n迭代 {iteration} 总耗时: {total_time:.2f} 秒")
    print(f"生成图谱时间: {graph_build_time:.2f} 秒")
    print(f"查询时间: {query_time:.2f} 秒")
    
    # 输出token统计
    print("\n迭代 {iteration} 各阶段Token使用统计:")
    print(f"1. 构建图谱阶段token数: {graph_tokens}")
    print(f"2. 查询提示词token数: {query_prompt_tokens}")
    print(f"3. 查询响应内容token数: {query_completion_tokens}")
    print(f"总token数: {graph_tokens + query_prompt_tokens + query_completion_tokens}")

    # 保存结果到文件
    if response is not None:
        response_text = str(response)        
        try:
            Path(answer_dir).mkdir(parents=True, exist_ok=True)
            with open(answer_path, "w", encoding="utf-8") as file:
                file.write(f"查询问题: {query}\n\n")
                file.write(response_text)
                file.write(f"\n\n总耗时: {total_time:.2f} 秒")
                file.write(f"\n生成图谱时间: {graph_build_time:.2f} 秒")
                file.write(f"\n查询时间: {query_time:.2f} 秒")
                file.write(f"\n\n各阶段Token使用统计:")
                file.write(f"\n1. 构建图谱阶段token数: {graph_tokens}")
                file.write(f"\n2. 查询提示词token数: {query_prompt_tokens}")
                file.write(f"\n3. 查询响应内容token数: {query_completion_tokens}")
                file.write(f"\n总token数: {graph_tokens + query_prompt_tokens + query_completion_tokens}")
            print(f"迭代 {iteration} 内容已写入: {answer_path}")
            
        except Exception as e:
            print(f"迭代 {iteration} 保存文件错误: {e}")
            return False
    else:
        print(f"警告: 迭代 {iteration} 查询失败，无内容保存。")
        return False

    # 备份dickens目录到data文件夹
    try:
        backup_dir = os.path.join(DATA_BACKUP_DIR, f"dickens_backup_{response_num}")
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(WORKING_DIR, backup_dir)
        print(f"已将dickens目录备份到: {backup_dir}")
    except Exception as e:
        print(f"备份dickens目录时出错: {e}")
    
    print(f"===== 第 {iteration} 次循环完成 =====")
    return True


async def main(loop_count=1):
    """主函数：从response11开始循环3次"""
    configure_logging()
    
    print(f"开始从response207到response210的{loop_count}次循环处理...")
    success_count = 0
    
    for i in range(1, loop_count + 1):
        iteration_success = await process_iteration(i)
        if iteration_success:
            success_count += 1
    
    print(f"\n循环处理完成！共执行 {loop_count} 次，成功 {success_count} 次。")


if __name__ == "__main__":
    asyncio.run(main(loop_count=1))
    print("\n所有任务执行完成!")
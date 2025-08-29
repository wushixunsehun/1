import asyncio
import os
import inspect
import logging
import logging.config
import time
from lightrag import LightRAG, QueryParam
from lightrag.utils import TokenTracker
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv
import json

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./dickens"
token_tracker = TokenTracker()


def configure_logging():
    """配置日志系统"""
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_ollama_demo.log"))
    print(f"\n日志文件: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s: %(message)s"},
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {"formatter": "default", "class": "logging.StreamHandler"},
            "file": {"formatter": "detailed", "class": "logging.handlers.RotatingFileHandler",
                     "filename": log_file_path, "maxBytes": log_max_bytes, "backupCount": log_backup_count, "encoding": "utf-8"},
        },
        "loggers": {"lightrag": {"handlers": ["console", "file"], "level": "INFO", "propagate": False}},
    })

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# 增强版计时器，支持记录时间并添加到全局统计
class Timer:
    def __init__(self, task_name, stats):
        self.task_name = task_name
        self.stats = stats  # 全局统计字典
        self.start_time = 0
        self.elapsed_time = 0

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.perf_counter() - self.start_time
        print(f"[计时] {self.task_name}: {self.elapsed_time:.4f} 秒")
        self.stats["total_time"] += self.elapsed_time
        self.stats[f"time_{self.task_name.replace(' ', '_')}"] = self.elapsed_time


# 获取Ollama模型的token数量并添加到全局统计
async def count_ollama_tokens(text, model_name, stats):
    """通过Ollama API获取文本的token数量并更新统计"""
    import aiohttp

    host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    url = f"{host}/api/tokenize"
    payload = {
        "model": model_name,
        "text": text
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                token_count = len(result.get("tokens", []))
            else:
                error_msg = await response.text()
                print(f"[警告] token计数失败: {error_msg}")
                token_count = len(text) // 4  # 简单估算

    print(f"[token统计] {text[:20]}...: {token_count}")
    stats["total_token"] += token_count
    stats[f"token_{text[:8].replace(' ', '_')}"] = token_count
    return token_count


async def initialize_rag(stats):
    llm_model = os.getenv("LLM_MODEL", "qwen3:1.7b")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": 8192},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    stats["llm_model"] = llm_model
    return rag


async def print_stream_with_stats(stream, query_text, model_name, stats):
    """打印流式响应并统计token数量，更新全局统计"""
    response_text = ""
    start_time = time.perf_counter()

    async for chunk in stream:
        print(chunk, end="", flush=True)
        response_text += chunk

    elapsed_time = time.perf_counter() - start_time
    token_count = await count_ollama_tokens(response_text, model_name, stats)

    print(f"\n[响应统计] 响应时间: {elapsed_time:.4f} 秒, 响应token: {token_count}")
    stats["time_response"] = elapsed_time
    return response_text


async def main():
    # 初始化全局统计字典
    stats = {
        "total_time": 0,
        "total_token": 0,
        "llm_model": ""
    }

    try:
        # 清理旧数据
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

        # 初始化RAG并计时
        async with Timer("初始化RAG", stats):
            rag = await initialize_rag(stats)

        # 测试嵌入功能并计时
        test_text = ["This is a test string for embedding."]
        async with Timer("嵌入测试", stats):
            embedding = await rag.embedding_func(test_text)
        
        # 统计嵌入测试的token
        embed_token = await count_ollama_tokens(test_text[0], stats["llm_model"], stats)

        print("\n=======================")
        print("嵌入功能测试")
        print("========================")
        print(f"测试文本: {test_text}")
        print(f"检测到嵌入维度: {embedding.shape[1]}\n\n")

        # 插入文档并计时
        with open("./file_response1/response1.txt", "r", encoding="utf-8") as f:
            doc_content = f.read()
            async with Timer(f"插入文档 (大小: {len(doc_content)} 字符)", stats):
                await rag.ainsert(doc_content)
            
            # 统计文档token
            doc_token = await count_ollama_tokens(doc_content, stats["llm_model"], stats)

        # 执行查询
        query = "Lustre 文件系统有哪些核心组件？请说明每个组件的作用。"
        async with Timer("查询处理", stats):
            # 统计查询token
            query_token = await count_ollama_tokens(query, stats["llm_model"], stats)
            
            print("\n=====================")
            print("查询模式: local")
            print("=====================")
            
            # 记录检索时间（假设aquery包含检索逻辑）
            start_retrieval = time.perf_counter()
            resp = await rag.aquery(
                query,
                param=QueryParam(mode="global", stream=True),
            )

            retrieval_time = time.perf_counter() - start_retrieval
            print(f"[计时] 检索时间: {retrieval_time:.4f} 秒")
            stats["time_retrieval"] = retrieval_time
            stats["total_time"] += retrieval_time

            if inspect.isasyncgen(resp):
                await print_stream_with_stats(resp, query, stats["llm_model"], stats)
            else:
                response_text = resp
                # 统计响应token
                response_token = await count_ollama_tokens(response_text, stats["llm_model"], stats)

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'rag' in locals() and rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()
            print("[系统] 存储已清理完成")
        
        # 打印最终综合指标
        print("\n=====================")
        print("最终综合指标")
        print("=====================")
        print(f"[总时间] {stats['total_time']:.4f} 秒")
        print(f"[总token] {stats['total_token']}")
        
        # 打印详细统计
        print("\n详细统计:")
        for key, value in stats.items():
            if key.startswith("time_") or key.startswith("token_"):
                display_name = key.replace("_", " ").title()
                print(f"{display_name}: {value}")


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\n全部流程完成")
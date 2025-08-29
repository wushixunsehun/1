import os
import sys
system_perception_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('./')
sys.path.append('../')
sys.path.append('.../')
sys.path.append('..../')


import logging
from agentsAPI import query_llm, get_rag_rpc
from elasticsearch import AsyncElasticsearch


logging.basicConfig(
    filename = os.path.join(system_perception_dir, "logs/system_perception_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """你是 Elasticsearch 查询方面的专家，擅长将用户的自然语言需求转化为 QueryDSL 查询语句。
请根据用户提出的查询目标、字段、过滤条件等要求，生成对应的 QueryDSL 查询语句。

用户查询：
{query}

要求：
- 只返回可以直接执行的 QueryDSL 查询语句
- 不需要附加任何说明性内容
"""


class LogdbExpert():
    """自然语言 -> DSL -> 访问 Elasticsearch 数据库 -> 返回日志数据"""
    def __init__(self, config):
        # 初始化 Elasticsearch 客户端
        self.es_endpoint = config["url"]
        self.username = config['username']
        self.password = config["password"]

        self.es = AsyncElasticsearch(
            hosts = [self.es_endpoint],
            basic_auth = (self.username, self.password)
        )


    def gen_dsl(self, query: str, source: str, need_rag: bool):
        """
        根据 query 执行 LLM 询问，返回可使用 HTTP API 访问 Elasticsearch 数据库的查询语句
        """
        context = get_rag_rpc(query, source) if need_rag else ""

        # 组合提示词、RAG、query
        prompt = PROMPT_TEMPLATE.format(query=query, context=context)

        dsl = query_llm(prompt)
        return dsl


    async def query_elasticsearch(es: AsyncElasticsearch, index: str, query_dsl: str):
        """
        向 ElasticSearch 日志数据库发送请求并返回结果
        """
        try:
            response = await es.search(index=index, body=query_dsl)
            return response
        except Exception as e:
            return {"error": str(e), "query": query_dsl, "index": index}
    

    async def get_logdb_state(self, query: str, index: str) -> dict:
        logger.info(f"输入查询：{query}")

        # 1. 自然语言转化为 QueryDSL
        source = ['']
        need_rag = True
        query_dsl = self.gen_dsl(query, source, need_rag)
        logger.info(f"生成 QueryDSL：{query_dsl}")


        # 2. 执行 QueryDSL 查询
        try:
            response_result = await self.query_elasticsearch(self.es, index, query_dsl)
            logger.info(f"Elasticsearch 数据库查询结果：{response_result}")
            return {"status": "success", "data": response_result.strip()}

        except Exception as e:
            logger.error(f"Elasticsearch 数据库查询失败：{str(e)}")
            return {"status": "error", "message": str(e)}


    async def get_logdb_state_stream(self, query: str, index: str):
        pass
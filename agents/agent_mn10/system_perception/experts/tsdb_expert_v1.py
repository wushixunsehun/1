import os
import sys
system_perception_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('./')
sys.path.append('../')
sys.path.append('.../')
sys.path.append('..../')


import logging
import requests
from typing import Optional
from urllib.parse import parse_qs
from urllib.parse import urlencode
from agentsAPI import query_llm, get_rag_rpc


logging.basicConfig(
    filename = os.path.join(system_perception_dir, "logs/system_perception_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """你是 Prometheus 查询领域的专家，擅长通过逐步推理的方式，将用户的特定需求转化为 PromQL 查询。
这包括了解用户的监控需求、他们感兴趣的具体指标，他们想要数据的时间段以及他们想要应用的任何特定条件或阈值。 
你的目标是根据给定的信息提供最准确、最高效的 PromQL 查询。

请以能与 HTTP API 调用的格式返回 PromQL，例如：
'query?query=up&time=2015-07-01T20:10:51.781Z'用于即时数据查询。
'query_range?query=up&start=2015-07-01T20:10:30.781Z&end=2015-07-01T20:11:00.781Z&step=15s'用于范围数据查询。

可能与用户查询相关的指标信息：
{context}

用户查询：
{query}

要求：
1.我会填写 Prometheus 的 IP 地址和端口号
2.你可以参考指标上下文中的信息，获取更准确的指标名称，优化你的查询。
3.禁止返回任何解释性内容，你只需要根据示例提供查询即可
4.输出应该是纯文本格式，不要包含任何代码块或 Markdown 格式。
"""

class TSdbExpert():
    """自然语言 -> PromQL -> 访问 Prometheus 数据库 -> 返回时序数据"""

    def __init__(self, config):
        self.prometheus_endpoint = config["url"]


    def gen_promql(self, query: str, source: list, need_rag: bool):
        """
        根据 query 执行 LLM 询问，返回可使用 HTTP API 访问 prometheus 数据库的查询语句
        """
        # 获取 RAG 文档信息
        context = get_rag_rpc(query, source) if need_rag else ''

        # 组合提示词、RAG、query
        prompt = PROMPT_TEMPLATE.format(query=query, context=context)

        promql = query_llm(prompt)
        return promql


    def get_type_and_para_from_promql(self, prometheus_endpoint: str, promql: str):
        """
        从 promql 字符串中提取 query_type 和参数，组合返回访问链接和参数
        """
        query_type, query_params = promql.split('?')
        prometheus_url = f"{prometheus_endpoint}/api/v1/{query_type}"
        params = {k: v[0] for k, v in parse_qs(query_params).items()}
        return prometheus_url, params


    def query_prometheus(self, prometheus_url: str, params: dict):
        """
        向 Prometheus 时序数据库发送请求并返回结果
        """
        # 只拼接有值的参数，避免空值导致的无效请求

        query_url = prometheus_url + '?' + urlencode(params)
        try:
            response = requests.get(query_url, timeout=100)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "url": query_url, "params": params}
    

    async def get_tsdb_state(self, query: str) -> dict:
        logger.info(f"输入查询：{query}")

        # 1. 获取指标上下文，结合上下文补全查询
        source = ['metric']
        need_rag = True
        promql = self.gen_promql(query, source, need_rag)
        logger.info(f"生成 PromQL：{promql}")

        # 2. 解析 PromQL 查询类型和参数
        try:
            prometheus_url, params = self.get_type_and_para_from_promql(self.prometheus_endpoint, promql)
        except Exception as e:
            logger.error(f"PromQL 解析失败：{str(e)}")
            return {"status": "error", "message": f"PromQL 解析失败: {str(e)}", "promql": promql}

        # 3. 查询 Prometheus 并返回数据
        try:
            response_result = self.query_prometheus(prometheus_url, params)
            logger.info(f"Prometheus 数据库查询结果：{response_result}")
            return {"status": "normal", "data": response_result, "promql": promql}
        except Exception as e:
            logger.error(f"Prometheus 数据库查询失败：{str(e)}")
            return {"status": "error", "message": str(e), "promql": promql}


    async def get_tsdb_state_stream(self, query: str):
        pass
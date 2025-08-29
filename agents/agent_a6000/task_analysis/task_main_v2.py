import os, re, sys
sys.path.append('./')
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)


import time, yaml
import json, uuid
import httpx, uvicorn
from mem0 import Memory
import logging, requests
from pathlib import Path
from typing import Optional
from functools import partial
from agent_state import AgentState
from fastapi import FastAPI, Request
from langgraph.graph import StateGraph, END
from agentsAPI import query_llm, strip_think
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from fastapi.responses import JSONResponse, StreamingResponse
from prompts import UPDATE_MEMORY_PROMPT, custom_fact_extraction_prompt


logging.basicConfig(
    filename = os.path.join(log_dir, "task_analysis_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.propagate = False


agents_dir = Path(__file__).resolve().parents[2]
config_path = agents_dir / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

llm_model = config.get("llm", {}).get("model")
llm_host = config.get("llm", {}).get("base_url")
embedding_model = config.get("rag", {}).get("embedding_model")
embedding_host = config.get("rag", {}).get("base_url")
api_key = config.get("rag", {}).get("api_key")

milvus_collection = config.get("collections", {}).get("mem0_col")
milvus_host = config.get("milvus", {}).get("uri")
milvus_token = config.get("milvus", {}).get("token")


mem0_config = {
    "version": "v1.1",
    "custom_fact_extraction_prompt": custom_fact_extraction_prompt,

    # --- 8< --- 本地 LLM 服务 --- 8< ---
    "llm": {
        "provider": "openai",
        "config": {
            "model": llm_model,
            "openai_base_url": llm_host,
            "api_key": api_key,
        }
    },

    # --- 8< --- 本地 Embedding 服务 --- 8< ---
    "embedder": {
        "provider": "openai",
        "config": {
            "model": embedding_model,
            "openai_base_url": embedding_host,
            "api_key": api_key,
            "embedding_dims": 768,
        }
    },

    # --- 8< --- Milvus 向量库 --- 8< ---
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": milvus_collection,
            "embedding_model_dims": 768,
            "url": milvus_host,
            "token": milvus_token,
        }
    },

    # --- 8< --- 自定义更新记忆提示词 --- 8< ---
    "prompts": {
        "update_memory": UPDATE_MEMORY_PROMPT,
    }
}

# mem0 = Memory.from_config(mem0_config)


class Node:
    def __init__(
        self,
        name: str,
        agent_type: str,
        next_nodes: list[str],
        hostname: Optional[str] = None,
        expert: Optional[str] = None,
        sub_task: Optional[str] = None,
    ):
        self.name = name
        self.agent_type = agent_type
        self.next_nodes = next_nodes
        self.hostname = hostname
        self.expert = expert
        self.sub_task = sub_task

    def add_next_node(self, next_node):
        self.next_nodes.append(next_node)

    def __repr__(self):
        return f"节点名称：{self.name}，类型：{self.agent_type}，下一跳：{self.next_nodes}"


TASK_PROMPT_TEMPLATE = '''你是一位专业的智能系统任务决策专家，请严格根据用户输入的任务描述进行意图分析，并精确匹配到以下任务类型：

### 任务类型定义
**知识问答任务**
- 特征：用户询问事实性/解释性问题。
- 系统行为：检索知识库(RAG)，整理信息并输出答案
- 示例：
    "如何安装 xx 软件/硬件？"
    "如何查看 xx 节点的状态？"

**状态查询任务**
- 特征：用户请求获取系统/服务的实时状态信息
- 系统行为：生成终端命令→执行操作→整理状态报告
- 示例：
    "检查 xx 服务的运行状态"
    "xx 作业的活跃时长是多少？"

**异常处理任务**
- 特征：用户报告故障/异常，需要诊断和解决
- 系统行为：获取系统状态→分析原因→执行修复→生成处理报告
- 示例：
    "磁盘出现报错，分析原因给出解决方案"
    "xx 任务运行中断，请检查原因"

**写操作任务**
- 特征：用户需要对系统资源进行“写”操作，如修改/创建文件、更新配置等
- 系统行为：基于任务描述直接调用符合需求的脚本/运维剧本→执行→返回结果确认
- 示例：
    "把 xx 中的 xx 调成 xx"
    "修改用户 xx 的作业配额"
    "取消作业 xx"

### 分析规则
1. 严格基于文本字面含义分析，禁止主观推测
2. 无状态操作的知识请求 ≠ 状态查询
3. 单纯询问解决方案 ≠ 异常处理（需实际故障描述）
4. 对系统资源的变更操作优先归类为写操作任务

### 输出要求
- 仅返回字符串形式的任务类型（知识问答任务、状态查询任务、异常处理任务、写操作任务）
- 禁止任何解释或附加文本

请分析任务：{task}
'''


def judge_task_type(task: str):
    prompt = TASK_PROMPT_TEMPLATE.format(task=task)

    enable_thinking = config.get("intent_recog_think")
    if enable_thinking:
        response = query_llm(prompt, enable_thinking=enable_thinking)
        response = strip_think(response).strip()
    else:
        response = query_llm(prompt)

    return response


def generate_dag_dict(task):
    node_dict = {}
    start_node = ""
    
    logger.info(f"输入任务:{task}")
    task_type = judge_task_type(task)
    # print(f"任务类型:{task_type}")
    logger.info(f"任务类型:{task_type}")


    if task_type == '知识问答任务':
        """
        操作执行任务：报告反馈Agent
        """
        node_dict["报告反馈"] = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname='a6000-G5500-V6')
        start_node = "报告反馈"


    if task_type == '状态查询任务':
        """
        状态查询任务：系统感知 Agent----（内置操作执行 Agent）----报告反馈 Agent
        """
        # 构建系统感知Agent节点
        node_dict["系统感知"] = Node("系统感知", "系统感知Agent", next_nodes=["报告反馈"], hostname='mn10')

        # 构建报告反馈Agent节点
        node_dict["报告反馈"] = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname='a6000-G5500-V6')
        start_node = "系统感知"


    if task_type == '异常处理任务':
        """
        异常处理任务：系统感知 Agent----异常分析 Agent----策略规划 Agent----操作执行 Agent----报告反馈 Agent
        """
        # 构建系统感知Agent节点
        node_dict["系统感知"] = Node("系统感知", "系统感知Agent", next_nodes=["异常分析"], hostname='mn10')

        # 构建异常分析Agent节点
        node_dict["异常分析"] = Node("异常分析", "异常分析Agent", next_nodes=["策略规划"], hostname='a6000-G5500-V6')

        # 构建策略规划Agent节点
        node_dict["策略规划"] = Node("策略规划", "策略规划Agent", next_nodes=["报告反馈"], hostname='mn10')

        # 构建报告反馈Agent节点
        node_dict["报告反馈"] = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname='a6000-G5500-V6')
        start_node = "系统感知"


    if task_type == '写操作任务':
        """
        写操作任务：策略规划 Agent----报告反馈 Agent
        """
        # 构建策略规划Agent节点
        node_dict["策略规划"] = Node("策略规划", "策略规划Agent", next_nodes=["报告反馈"], hostname='mn10')

        # 构建报告反馈Agent节点
        node_dict["报告反馈"] = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname='a6000-G5500-V6')
        start_node = "策略规划"

    return node_dict, start_node


def travel_dag(nodes, stack):
    if stack[0] == 'END':
        return
    node = nodes[stack[0]]
    print(node.name)
    print("  |")
    stack.pop(0)
    for item in node.next_nodes:
        if item not in stack:
            stack.append(item)
    travel_dag(nodes, stack)

    return


def print_dag(nodes, start_node):
    print("生成DAG流:")
    travel_dag(nodes, [start_node])
    print(" END")


def node_func(state: AgentState):
    print("测试缓冲节点......")
    return


def report_memory_changes(mem_results: list[dict]) -> None:
    """
    把 Mem0 返回的结果分组打印到终端和日志。
    只处理 event≠NONE 的条目；完全无变动时静默。
    """
    changes = {"ADD": [], "UPDATE": [], "DELETE": []}

    for r in mem_results:
        evt = r.get("event", "NONE")
        if evt != "NONE":
            changes[evt].append(r)

    if not any(changes.values()):
        return  # 没有任何新增 / 更新 / 删除

    lines: list[str] = []
    if changes["ADD"]:
        lines.append("🔹 新增记忆:")
        lines.extend([f'  + {m["memory"]}' for m in changes["ADD"]])

    if changes["UPDATE"]:
        lines.append("🔸 更新记忆:")
        lines.extend(
            [
                f'  ~ {m["old_memory"]}  →  {m["memory"]}'
                if "old_memory" in m
                else f'  ~ {m["memory"]}'
                for m in changes["UPDATE"]
            ]
        )

    if changes["DELETE"]:
        lines.append("🔻 删除记忆:")
        lines.extend([f'  - {m["memory"]}' for m in changes["DELETE"]])

    msg_out = "\n".join(lines)
    # print(f'\n{msg_out}\n')
    logger.info("\n" + msg_out + "\n")


def result2reply_text(result):
    if isinstance(result, str):
        reply_text = result
    elif isinstance(result, dict):
        # 兼容原有逻辑
        if "result" in result and isinstance(result["result"], str):
            reply_text = result["result"]
        elif "result" in result and isinstance(result["result"], dict):
            # 原有的 result['result']['parts'][0]['text'] 结构
            parts = result["result"].get("parts")
            if parts and isinstance(parts, list) and "text" in parts[0]:
                reply_text = parts[0]["text"]
            else:
                reply_text = str(result)
        # 新增：兼容 {'0': {'root_cause': ...}} 结构
        elif all(isinstance(v, dict) and "root_cause" in v for v in result.values()):
            # 只取第一个key
            first_key = next(iter(result))
            v = result[first_key]
            reply_text = (
                f"根因节点: {v.get('root_cause', '')}\n"
                f"故障类型: {v.get('failure_type', '')}\n"
                f"Top5根因: {v.get('top5', '')}"
            )
        else:
            reply_text = str(result)
    elif isinstance(result, list):
        reply_text = f'指标时序异常点: {result}'
    else:
        reply_text = str(result)
    return reply_text


def push_user_turn(state: AgentState, hostname: str, task: str):
    if task:
        key = ("u", hostname, task)
        if key not in state.get("_msg_keys", set()):
            msg = {
                "role": "user",
                "hostname": hostname,
                "task": task,
            }
            state["messages"] = state.get("messages", []) + [msg]
            state.setdefault("_msg_keys", set()).add(key)
            # logger.info(f"添加条目：{msg}")


def push_agent_turn(state: AgentState, hostname: str, agent_name: str, response: str):
    if response:
        key = ("a", hostname, agent_name, response)
        msg = {
            "role": "agent",
            "hostname": hostname,
            "name": agent_name,
            "response": response,
        }
        state["messages"] = state.get("messages", []) + [msg]
        state["_msg_keys"].add(key)
        # logger.info(f"添加条目：{msg}")


def build_agent_url(agent_type: str, hostname: str) -> str:
    """
    根据 agent_type 和 hostname 拼接 URL。
    端口与路径在此集中管理，后期要换端口只改这里即可。
    """
    route_config = {
        "system_perception": ("5001", "system_perception"),
        "anomaly_analysis": ("5002", "anomaly_analysis"),
        "strategy_plan": ("5003", "strategy_plan"),
        "command_run": ("5004", "command_run"),
        "report_generate": ("5005", "report_generate"),
    }

    if agent_type not in route_config:
        raise KeyError(f"unknown agent_type {agent_type}")

    port, path = route_config[agent_type]
    return f"http://{hostname}:{port}"


async def agent_quest(agent_type: str, node: Node, query: str):
    """
    采用异步通信处理http交互，  任务解析 —— 其他 agent —— 大模型 api
    """
    hostname = node.hostname or node.name.split("_")[0]
    if hostname in {"start", "end"}:
        return ""

    base_url = build_agent_url(agent_type, hostname)

    # mem_msg = [
    #     {"role": "user", "content": query},
    #     {"role": "assistant", "content": ''}
    # ]

    # mem_res = mem0.add(
    #     mem_msg,
    #     user_id = 'agent',
    #     metadata = {"source": "chat", "time": time.time()}
    # )
    # report_memory_changes(mem_res.get("results", []))

    async with httpx.AsyncClient(verify=False, timeout=300, proxy=None) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url,)
        final_agent_card_to_use = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)

        # mems = mem0.search(query, user_id='agent', limit=5)
        # mem_context = "\n".join(f"- {r['memory']}" for r in mems["results"])

        # system_prompt = f"{query}\n以下是与用户相关的记忆：\n{mem_context}\n请回答"

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": query}],
                "messageId": uuid.uuid4().hex,
            },
        }
        logger.info(f"📤 Sending request to {agent_type}, url: {base_url}")

        # return collected_text
        request = SendMessageRequest(
            id=str(uuid.uuid4()), params=MessageSendParams(**payload)
        )

        response = await client.send_message(request)
        logger.info(f"📥 Response received from {agent_type}, url: {base_url}")

        resp_result = response.model_dump(mode='json', exclude_none=True)

        return resp_result


async def system_perception(state: AgentState, node: Node):
    """
    系统感知 agent，负责收集系统状态信息
    """
    agent = "system_perception"
    # print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    # 从节点中提取相关信息
    hostname = node.hostname
    task = state['query']
    expert = node.expert

    if task:
        push_user_turn(state, hostname, task)

    agent_result = await agent_quest(agent, node, task)

    # 将结果转换为文本格式
    reply_text = result2reply_text(agent_result)
    reply_text = json.loads(reply_text)["result"]

    # 将 agent 的响应记录到状态
    push_agent_turn(state, hostname, node.agent_type, reply_text)
    # logger.info(f"🤖 Agent {agent} 响应: {str(json.loads(reply_text)['state_exp'])}")

    # 返回差分结果，更新状态
    return {
        "hostname": hostname,
        "sub_task": '',
        "expert": expert,
        "result": {agent: reply_text},
        "status": "success",
    }


async def anomaly_analysis(state: AgentState, node: Node):
    """
    异常分析 agent，根据模型分析结果来定位异常根因
    """
    # return state
    agent = "anomaly_analysis"
    # print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    hostname  = node.hostname
    sub_task  = state['query']

    if sub_task:
        push_user_turn(state, hostname, sub_task)

    # 调用 agent
    if "指标" in sub_task or "异常" in sub_task:
        payload = {
            "folder_path": "/home/tanxh/mas/agents/anomaly_model/AnomalyDetection/data/Dataset1/Node1.csv"
        }
        url = f"http://localhost:5411/time_series_ad"
        response = requests.post(url, json=payload)
        agent_result = response.json()
    elif "根因定位" in sub_task:
        payload = {
            "folder_path": "/home/tanxh/mas/agents/anomaly_model/HPC_RCA_Demo/dataset",
            "job_id": "job_8113170",
            "failure_start": 1750129350,
            "failure_end": 1750129800,
            "failure_type": "network_bandwidth",
            "root_node": "cn61903",
            "golden_metrics": [],
            "top_k": 5
        }
        url = f"http://localhost:5410/locate_root_cause"
        response = requests.post(url, json=payload)
        agent_result = response.json()
    else:
        agent_result = "no reply"

    reply_text = result2reply_text(agent_result)

    # 记录响应
    push_agent_turn(state, hostname, node.agent_type, reply_text)
    # logger.info(f"🤖 Agent {agent} 响应: {str(reply_text)}")

    return {
        "hostname": hostname,
        "sub_task": '',
        "result": {agent: reply_text},
        "status": "success",
    }


async def strategy_plan(state: AgentState, node: Node):
    """
    策略规划 agent，通过上游节点的结果来生成修复策略
    """
    agent = "strategy_plan"
    logger.info(f"🤖 Agent {agent} 执行中...")

    hostname = node.hostname
    query = state['query']
    up_process_result = state.get("result") if state.get("result") else state.get('query')
    confirmed_params = state.get("confirmed_params")

    if isinstance(up_process_result, dict):
        full_report_lines = []
        for agent_name, raw in up_process_result.items():
            if isinstance(raw, str):
                if raw.startswith("根因节点:") or ("root_cause" in raw and "故障类型" in raw):
                    full_report_lines.append(f"## Agent: anomaly_analysis")
                    full_report_lines.append("```")
                    full_report_lines.append(raw.strip())
                    full_report_lines.append("```")
                    full_report_lines.append("")
                    continue
                if raw.startswith("指标时序") or ("异常点" in raw):
                    full_report_lines.append(f"## Agent: anomaly_analysis")
                    full_report_lines.append("```")
                    full_report_lines.append(raw.strip())
                    full_report_lines.append("```")
                    full_report_lines.append("")
                    continue
            text = raw.replace("\\n", "\n").replace("\\t", "\t")
            blocks = re.split(r'(?=【[^】]+】)', text)
            for block in blocks:
                m = re.match(r'【([^】]+)】\s*(.*)', block, re.DOTALL)
                if not m:
                    continue
                sub_host, rest = m.group(1), m.group(2).strip()
                full_report_lines.append(f"## Agent: {agent_name}")
                full_report_lines.append(f"### {sub_host}")
                full_report_lines.append("```")
                full_report_lines.append(rest)
                full_report_lines.append("```")
                full_report_lines.append("")
        all_up_res = "\n".join(full_report_lines)
    elif isinstance(up_process_result, str):
        all_up_res = up_process_result

    anomaly_desc = f"# {query}\n{all_up_res}"
    push_user_turn(state, hostname, anomaly_desc)

    # 构造请求数据
    request_data = {
        "query": anomaly_desc,
        "confirmed_params": confirmed_params
    }
    agent_result = await agent_quest(agent, node, json.dumps(request_data))
    result = json.loads(result2reply_text(agent_result))

    # 参数确认流程
    if result.get("status") == "need_confirm":
        # 返回给客户端，等待参数确认
        return {
            "hostname": hostname,
            "result": {agent: None},
            "status": "need_params_confirm",
            "playbooks": result.get("playbooks"),
            "query": result.get("query"),
        }

    # 正常执行流程
    reply_text = result.get("result", "")
    push_agent_turn(state, hostname, node.agent_type, reply_text)
    return {
        "hostname": hostname,
        "result": {agent: reply_text},
        "status": "success",
    }


async def report_generate(state: AgentState, node: Node):
    """
    报告反馈 agent，负责生成最终运维报告
    """
    agent = "report_generate"
    # print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    hostname  = node.hostname
    results = state.get("result") if state.get("result") else state.get('query')

    if isinstance(results, dict):
        full_report_lines = []

        for agent_name, raw in results.items():
            # 1. 尝试解析为 JSON 并提取 state_exp
            if isinstance(raw, str):
                # 判断是否为根因分析格式
                if raw.startswith("根因节点:") or ("root_cause" in raw and "故障类型" in raw):
                    # 直接输出
                    full_report_lines.append(f"## Agent: anomaly_analysis")
                    full_report_lines.append("```")
                    full_report_lines.append(raw.strip())
                    full_report_lines.append("```")
                    full_report_lines.append("")
                    continue

                if raw.startswith("指标时序") or ("异常点" in raw):
                    # 直接输出
                    full_report_lines.append(f"## Agent: anomaly_analysis")
                    full_report_lines.append("```")
                    full_report_lines.append(raw.strip())
                    full_report_lines.append("```")
                    full_report_lines.append("")
                    continue

            # 3. 默认按 state_exp 格式处理
            # 解码转义字符
            text = raw.replace("\\n", "\n").replace("\\t", "\t")
            # 分割每个主机块
            blocks = re.split(r'(?=【[^】]+】)', text)
            for block in blocks:
                m = re.match(r'【([^】]+)】\s*(.*)', block, re.DOTALL)
                if not m:
                    continue
                sub_host, rest = m.group(1), m.group(2).strip()
                full_report_lines.append(f"## Agent: {agent_name}")
                full_report_lines.append(f"### {sub_host}")
                full_report_lines.append("```")
                full_report_lines.append(rest)
                full_report_lines.append("```")
                full_report_lines.append("")

        full_report_lines.append(state['query'])
        full_report = "\n".join(full_report_lines)

    elif isinstance(results, str):
        full_report = results

    push_user_turn(state, hostname, full_report)

    # 调用 agent
    result = await agent_quest(agent, node, full_report)

    reply_text = result2reply_text(result)
    reply_text = json.loads(reply_text)["result"]

    # 记录响应
    push_agent_turn(state, hostname, node.agent_type, reply_text)
    # logger.info(f"🤖 Agent {agent} 响应: {str(reply_text)}")

    return {
        "hostname": hostname,
        "result": {agent: reply_text},
        "status": "success",
    }


# agent调用索引，按照节点名称配置合适的节点执行函数
agent_dict = {"系统感知Agent": system_perception, "异常分析Agent": anomaly_analysis, "策略规划Agent": strategy_plan,
            "报告反馈Agent": report_generate}


class WorkGraph:
    def __init__(self, dag, start):
        self.graph = None
        self.dag = dag
        self.start = start


    def create_fallback_judge_branch(self, workflow: StateGraph, pre_node, fallback_node):
        # 设置异常判断边
        def fallback_judge(state: AgentState):
            return "error" if state.get("status") == "error" else "success"

        dag_next_node = self.dag[pre_node].next_nodes[0]
        # 如果下一跳是字符串 'END'，就换成常量 END
        dag_next_node = END if dag_next_node == "END" else dag_next_node

        workflow.add_conditional_edges(
            pre_node,
            fallback_judge,
            {"error": fallback_node, "success": dag_next_node},
            # {"error": fallback_node, "success": next_node_name_or_END},
        )
        # print(f"为节点{pre_node}创建异常判断分支")


    def create_fallback_node(self, workflow):
        """
        遇故障自动切换处理策略,
        设置一个独立的异常处理节点,每个节点运行后使用条件边判断异常再交付
        """

        def fall_back(state: AgentState):
            """
            异常处理函数，可扩展增添其他功能
            """
            # print("异常处理节点")
            # print("系统状态:", state["status"])
            # return
            push_agent_turn(state, "fallback_node",
                    f"已触发回退，原因：{state['error_message']}")
            return {"status": "success"}

        workflow.add_node("fallback_node", fall_back)
        workflow.add_edge("fallback_node", END)
        # print("异常处理节点创建完成")


    def dag_to_langgraph(self):
        workflow = StateGraph(AgentState)

        # 初始化异常处理节点
        self.create_fallback_node(workflow)
        for node in self.dag.values():
            workflow.add_node(node.name, partial(agent_dict[node.agent_type], node=node))
            self.create_fallback_judge_branch(workflow, node.name, "fallback_node")
            # print(f"节点{node.name}创建成功")

        for node in self.dag.values():
            next_nodes = node.next_nodes
            for next_node_name in next_nodes:
                if next_node_name == 'END':
                    workflow.add_edge(node.name, END)
                else:
                    workflow.add_edge(node.name, next_node_name)

        # 设置入口
        workflow.set_entry_point(self.start)

        # 编译图
        self.graph = workflow.compile()


    def draw_graph(self):
        output_path = "agent_a6000/task_analysis/graph_figs/workflow_dag.png"

        self.graph.get_graph().draw_png(
            output_file_path = output_path,
            fontname = "Noto Sans CJK SC",
        )


    async def run_workflow(self, task_or_state):
        """
        支持两种调用方式：
        - 首次调用传入 task 字符串，自动构造初始 state
        - 参数确认后传入补充参数的 state dict，继续推进 DAG
        """
        if isinstance(task_or_state, dict):
            state = task_or_state
        else:
            state = {
                "messages": [],
                "query": task_or_state,
                "result": {},
                "status": 'success',
                "error_code": 0,
                "error_message": '',
                "hostname": None,
                "sub_task": None,
                "expert": None,
                "session_id": str(uuid.uuid4()),
            }

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # 用 astream 逐步推进，每步检查 status
        async for step_state in self.graph.astream(state, config):
            # 参数确认流程，主动暂停
            if step_state.get("status") == "need_params_confirm":
                payload = step_state
                try:
                    result = json.dumps(payload, ensure_ascii=False, indent=2)
                except TypeError:
                    result = str(payload)
                return result
        # 正常流程，最终返回
        payload = step_state["result"]
        try:
            result = json.dumps(payload, ensure_ascii=False, indent=2)
        except TypeError:
            result = str(payload)
        return result


def process_resp(result_str):
    resp = json.loads(result_str)

    final_key = "report_generate"
    if final_key not in resp:
        raise ValueError(f"缺少最终报告")

    # 1. 优化主机命令输出格式
    cmd_md_parts = []
    for agent, md in resp.items():
        if agent == final_key:
            continue
        if isinstance(md, str):
            blocks = re.split(r'(?=【[^】]+】)', md)
            for block in blocks:
                m = re.match(r'【([^】]+)】\s*(.*)', block, re.DOTALL)
                if not m:
                    continue
                sub_host, rest = m.group(1), m.group(2).strip()
                cmd_md_parts.append(f"## {sub_host}")
                cmd_md_parts.append("")
                cmd_md_parts.append("```bash")
                cmd_md_parts.append(rest)
                cmd_md_parts.append("```")
                cmd_md_parts.append("")
        else:
            cmd_md_parts.append(str(md))
    cmds_result = "\n".join(cmd_md_parts)

    # 2. 优化最终报告格式（去除多余代码块标记）
    final_md = resp[final_key].strip()
    final_md = re.sub(r'^\s*```markdown\s*', '', final_md)
    final_md = re.sub(r'\s*```\s*$', '', final_md)

    final_return = f"{cmds_result}\n{final_md}".strip()

    return JSONResponse({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": final_return
                }
            }
        ]
    })


# === OpenWebUI 风格 API 集成 ===
openwebui_app = FastAPI()


@openwebui_app.get("/v1/models")
async def get_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "MAS",  # 这个必须和 chat 请求中的 model 保持一致
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    })


@openwebui_app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    # 判断是否为参数补充（带 status/playbooks/confirmed_params/query 字段）
    is_param_resume = any(k in data for k in ["status", "playbooks", "confirmed_params", "query"]) and data.get("status") == "need_params_confirm"

    # 全局 session_id -> (dag, start) 缓存
    if not hasattr(chat_completions, "session_cache"):
        chat_completions.session_cache = {}
    session_cache = chat_completions.session_cache

    try:
        if is_param_resume:
            # 参数补充，直接用 state 继续推进
            session_id = data.get("session_id") or data.get("sessiong_id") or data.get("sessionid") or (data.get("state") and data["state"].get("session_id")) or data.get("state", {}).get("sessiong_id")
            if not session_id:
                session_id = data.get("state", {}).get("session_id")
            if not session_id or session_id not in session_cache:
                return JSONResponse({"error": "参数补充请求缺少 session_id 或 session_id 无效。"}, status_code=400)
            dag, start = session_cache[session_id]
            # 强制将 session_id 写入 state，保证后续节点都能访问
            data["session_id"] = session_id
            graph = WorkGraph(dag, start)
            graph.dag_to_langgraph()
            result = await graph.run_workflow(data)
        else:
            # 首次访问，生成 DAG
            user_query = ""
            if "messages" in data and isinstance(data["messages"], list):
                user_query = data["messages"][-1].get("content", "")
            elif "prompt" in data:
                user_query = data["prompt"]
            else:
                user_query = data.get("query", "")
            if not user_query:
                return JSONResponse({"error": "No user query provided."}, status_code=400)
            
            node_dict, start = generate_dag_dict(user_query)
            graph = WorkGraph(node_dict, start)
            graph.dag_to_langgraph()
            graph.draw_graph()
            session_id = str(uuid.uuid4())
            session_cache[session_id] = (node_dict, start)

            # 首次构造 state 时写入 session_id
            initial_state = {
                "messages": [],
                "query": user_query,
                "result": {},
                "status": 'success',
                "error_code": 0,
                "error_message": '',
                "hostname": None,
                "sub_task": None,
                "expert": None,
                "session_id": session_id
            }
            result = await graph.run_workflow(initial_state)

            # 在参数确认时，把 session_id 一并返回
            result_dict = json.loads(result)
            if result_dict.get("status") == "need_params_confirm":
                result_dict["session_id"] = session_id
                return JSONResponse(result_dict)

            return process_resp(result)
    except Exception as e:
        logger.exception("Agent workflow error:")
        return JSONResponse({"error": str(e)}, status_code=500)


# === 启动 FastAPI ===
if __name__ == '__main__':
    uvicorn.run(openwebui_app, host="0.0.0.0", port=5000)

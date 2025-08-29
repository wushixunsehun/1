import os
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('.../')
task_analysis_dir = os.path.dirname(os.path.abspath(__file__))


import json
import time
import uuid
import httpx
import asyncio
import uvicorn
import logging
from pathlib import Path
from functools import partial
from fastapi import FastAPI, Request
from langgraph.graph import StateGraph, END
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from fastapi.responses import JSONResponse, Response
from agentsAPI import query_llm, AgentState, jsonrpc_request, Node


logging.basicConfig(
    filename = os.path.join(task_analysis_dir, "logs/task_analysis_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def judge_task_type(task: str):
    PROMPT_TEMPLATE = '''你是一位专业的智能系统任务决策师，任务是根据用户提供的任务描述，准确判断任务类型并输出结果。现在，请严格按照以下要求完成任务分类：

    任务类型及关键特征：
    - **状态查询任务**：查询服务器状态 -> 整理数据 -> 生成报告
    - **异常处理任务**：获取服务器状态和数据库数据 -> 分析异常 -> 制定解决方案 -> 执行解决方案 -> 生成报告
    - **操作执行任务**：获取服务器状态 -> 规划执行策略 -> 执行操作 -> 生成报告
    - **知识问答任务**：接收问题 -> 检索知识库 -> 组织答案 -> 输出结果

    工作流程:
    1. 仔细分析用户提供的任务描述，提取关键信息。
    2. 根据任务描述的关键特征，与所有任务类型的标准进行比对。
    3. 确定任务所属类型。

    要求:
    - 准确判断用户所描述的任务属于状态查询任务、异常处理任务、操作执行任务还是知识问答任务。
    - 仅根据用户提供的任务描述进行判断，不引入额外的主观假设。
    - 确保判断结果清晰明确，便于用户理解和后续操作。

    接下来，根据任务：{task}，直接输出字符串格式的任务类型。
    '''

    prompt = PROMPT_TEMPLATE.format(task=task)
    answer = query_llm(prompt)
    return answer


def judge_num_of_object(task: str) -> dict[str, dict]:
    PROMPT_TEMPLATE = """
    你是系统感知调度器。解析任务: {task}
    对每台服务器输出 JSON，key 为服务器名，value 为:
    - sub_task : 该服务器要做什么
    - expert : 需要的专家类型 (state：通过 shell 指令获取服务器状态；tsdb：通过 promql 查询时序数据库数据；logdb：通过 dsl 查询日志数据库数据)

    要求：
    1. 每台服务器的子任务可能相同，也可能不同，你需要仔细分析任务来自行判断。
    2. 在测试阶段，我们假设都使用 state 专家，你需要合理的拆解子任务。
    3. 只返回 JSON。
    """

    prompt = PROMPT_TEMPLATE.format(task=task)
    raw = query_llm(prompt)
    data = json.loads(raw)
    for obj, meta in data.items():
        if "sub_task" not in meta or "expert" not in meta:
            raise ValueError(f"{obj} 缺少字段")
    return data


def generate_dag_dict(task):
    node_dict = {}
    start_node = ""
    task_type = judge_task_type(task)
    print(f"任务类型:{task_type}")
    logger.info(f"任务类型:{task_type}")


    if task_type == '状态查询任务':
        """
        状态查询任务：系统感知Agent----报告反馈Agent
        """
        objects = judge_num_of_object(task)
        print(f"任务处理对象:{objects}")
        logger.info(f"任务处理对象:{objects}")

        # 构建系统感知Agent节点
        node_start = Node("start_系统感知", "系统感知Agent", next_nodes=[])
        node_dict["start_系统感知"] = node_start

        for obj, meta in objects.items():
            node_name = f"{obj}_系统感知"
            node_start.add_next_node(node_name)
            node_dict[node_name] = Node(
                name = node_name,
                agent_type = "系统感知Agent",
                next_nodes = ["end_系统感知"],
                hostname = obj,
                expert = meta['expert'],
                sub_task = meta["sub_task"],
            )

        node_end = Node("end_系统感知", "系统感知Agent", next_nodes=["报告反馈"])
        node_dict["end_系统感知"] = node_end

        # 构建报告反馈Agent节点
        node_start = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname = 'a6000-G5500-V6')
        node_dict["报告反馈"] = node_start
        start_node = "start_系统感知"


    if task_type == '异常处理任务':
        """
        异常处理任务：系统感知Agent----异常分析Agent----策略规划Agent----操作执行Agent----报告反馈Agent
        """
        objects = judge_num_of_object(task)
        print(f"任务处理对象:{objects}")
        logger.info(f"任务处理对象:{objects}")
        # object_list = []

        # 构建系统感知Agent节点
        node_start = Node("start_系统感知", "系统感知Agent", next_nodes=[])
        node_dict["start_系统感知"] = node_start
        
        for obj, meta in objects.items():
            node_name = f"{obj}_系统感知"
            node_start.add_next_node(node_name)
            node_dict[node_name] = Node(
                name = node_name,
                agent_type = "系统感知Agent",
                next_nodes = ["end_系统感知"],
                hostname = obj,
                expert = meta['expert'],
                sub_task = meta["sub_task"],
            )

        node_end = Node("end_系统感知", "系统感知Agent", next_nodes=["异常分析"])
        node_dict["end_系统感知"] = node_end

        # 构建异常分析Agent节点
        node_dict["异常分析"] = Node("异常分析", "异常分析Agent", next_nodes=["策略规划"], hostname = 'a6000-G5500-V6')

        # 构建策略规划Agent节点
        node_dict["策略规划"] = Node("策略规划", "策略规划Agent", next_nodes=["start_操作执行"], hostname = 'a6000-G5500-V6')

        # 构建操作执行Agent节点
        node_start = Node("start_操作执行", "操作执行Agent", next_nodes=[])
        node_dict["start_操作执行"] = node_start

        for obj, meta in objects.items():
            node_name = f"{obj}_操作执行"
            node_start.add_next_node(node_name)
            node_dict[node_name] = Node(
                name = node_name,
                agent_type = "操作执行Agent",
                next_nodes = ["end_操作执行"],
                hostname = obj,
            )

        node_end = Node("end_操作执行", "操作执行Agent", next_nodes=["报告反馈"])
        node_dict["end_操作执行"] = node_end

        # 构建报告反馈Agent节点
        node_start = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname = 'a6000-G5500-V6')
        node_dict["报告反馈"] = node_start
        start_node = "start_系统感知"


    if task_type == '操作执行任务':
        """
        操作执行任务：系统感知Agent----策略规划Agent----操作执行Agent----报告反馈Agent
        """
        objects = judge_num_of_object(task)
        print(f"任务处理对象:{objects}")
        logger.info(f"任务处理对象:{objects}")

        # 构建系统感知Agent节点
        node_start = Node("start_系统感知", "系统感知Agent", next_nodes=[])
        node_dict["start_系统感知"] = node_start

        for obj, meta in objects.items():
            node_name = f"{obj}_系统感知"
            node_start.add_next_node(node_name)
            node_dict[node_name] = Node(
                name = node_name,
                agent_type = "系统感知Agent",
                next_nodes = ["end_系统感知"],
                hostname = obj,
                expert = meta['expert'],
                sub_task = meta["sub_task"],
            )

        node_end = Node("end_系统感知", "系统感知Agent", next_nodes=["策略规划"])
        node_dict["end_系统感知"] = node_end

        # 构建策略规划Agent节点
        node_dict["策略规划"] = Node("策略规划", "策略规划Agent", next_nodes=["start_操作执行"], hostname = 'a6000-G5500-V6')

        # 构建操作执行Agent节点
        node_start = Node("start_操作执行", "操作执行Agent", next_nodes=[])
        node_dict["start_操作执行"] = node_start

        for obj, meta in objects.items():
            node_name = f"{obj}_操作执行"
            node_start.add_next_node(node_name)
            node_dict[node_name] = Node(
                name = node_name,
                agent_type = "操作执行Agent",
                next_nodes = ["end_操作执行"],
                hostname = obj,
            )

        node_end = Node("end_操作执行", "操作执行Agent", next_nodes=["报告反馈"])
        node_dict["end_操作执行"] = node_end

        # 构建报告反馈Agent节点
        node_start = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname = 'a6000-G5500-V6')
        node_dict["报告反馈"] = node_start
        start_node = "start_系统感知"


    if task_type == '知识问答任务':
        """
        操作执行任务：报告反馈Agent
        """
        print("知识问答没有对象")
        logger.info("知识问答没有对象")
        node_dict["报告反馈"] = Node("报告反馈", "报告反馈Agent", next_nodes=["END"], hostname = 'a6000-G5500-V6')
        start_node = "报告反馈"

    return [node_dict, start_node]


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


# agent 通信的相关参数与方法
# system_perception_url = 'http://localhost:5001/system_perception'
# anomaly_analysis_url = 'http://localhost:5002/anomaly_analysis'
# strategy_plan_url = 'http://localhost:5003/strategy_plan'
# command_run_url = 'http://localhost:5004/command_run'
# report_generate_url = 'http://localhost:5005/report_generate'


def node_func(state: AgentState):
    print("测试缓冲节点......")
    return


def result2reply_text(result):
    if isinstance(result, str):
        reply_text = result

    elif isinstance(result, dict):
        if "result" in result and isinstance(result["result"], str):
            reply_text = result["result"]

        else:
            reply_text = next(iter(result["result"].values()))

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
            logger.info(f"添加条目：{msg}")


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
        logger.info(f"添加条目：{msg}")


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
    return f"http://{hostname}:{port}/{path}"


async def agent_quest(agent_type: str, node: Node, payload: dict):
    """
    采用异步通信处理http交互，  任务解析——其他agent——大模型api
    data:dict
    """
    hostname = node.hostname or node.name.split("_")[0]
    if hostname in {"start", "end"}:
        return ""

    url = build_agent_url(agent_type, hostname)
    # url = 'http://localhost:5001/system_perception'

    async with httpx.AsyncClient() as client:
        try:
            # data = json.dumps(payload)
            response = await client.post(url, json=payload, timeout=3600)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            # 处理B服务返回的错误
            return {"error": f"service error: {exc.response.text}"}


async def system_perception(state: AgentState, node: Node):
    """
    系统感知 agent，根据当前节点类型来调用分机部署的agent
    """
    agent = "system_perception"
    print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    # 从节点中提取相关信息
    hostname = node.hostname
    sub_task = node.sub_task
    expert = node.expert

    # 如果有子任务，将其推送到状态中
    if sub_task:
        push_user_turn(state, hostname, sub_task)

    # 构造 JSON-RPC 2.0 请求的 payload，异步调用 agent
    payload = jsonrpc_request(state, agent, node)
    result = await agent_quest(agent, node, payload)

    # 将结果转换为文本格式
    reply_text = result2reply_text(result)

    # 将 agent 的响应记录到状态
    push_agent_turn(state, hostname, node.agent_type, reply_text)

    # 返回差分结果，更新状态
    return {
        "hostname": hostname,
        "sub_task": sub_task,
        "expert": expert,
        "result": {hostname: reply_text},
        "status": "success",
    }


async def anomaly_analysis(state: AgentState, node: Node):
    """
    节点执行函数，根据当前节点类型来调用分机部署的agent
    """
    pass
    agent = "anomaly_analysis"
    print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    hostname  = node.hostname
    sub_task  = node.sub_task
    
    if sub_task:
        push_user_turn(state, hostname, sub_task)

    # 调用 agent
    payload = jsonrpc_request(state, agent, node)
    result = await agent_quest(agent, node, payload)

    reply_text = result2reply_text(result)

    # 记录响应
    push_agent_turn(state, hostname, node.agent_type, reply_text)

    return {
        "hostname": hostname,
        "sub_task": sub_task,
        "result": {hostname: reply_text},
        "status": "success",
    }


async def strategy_plan(state: AgentState, node: Node):
    """
    节点执行函数，根据当前节点类型来调用分机部署的agent
    """
    pass
    agent = "strategy_plan"
    print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    hostname  = node.hostname
    sub_task  = node.sub_task
    
    if sub_task:
        push_user_turn(state, hostname, sub_task)

    # 调用 agent
    payload = jsonrpc_request(state, agent, node)
    result = await agent_quest(agent, node, payload)

    reply_text = result2reply_text(result)

    # 记录响应
    push_agent_turn(state, hostname, node.agent_type, reply_text)

    return {
        "hostname": hostname,
        "sub_task": sub_task,
        "result": {hostname: reply_text},
        "status": "success",
    }


async def command_run(state: AgentState, node: Node):
    """
    节点执行函数，根据当前节点类型来调用分机部署的agent
    """
    # 要发送的数据JSON-RPC 2.0规范
    agent = "command_run"
    print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    hostname  = node.hostname
    sub_task  = node.sub_task
    
    if sub_task:
        push_user_turn(state, hostname, sub_task)

    # 调用 agent
    payload = jsonrpc_request(state, agent, node)
    result = await agent_quest(agent, node, payload)

    reply_text = result2reply_text(result)

    # 记录响应
    push_agent_turn(state, hostname, node.agent_type, reply_text)

    return {
        "hostname": hostname,
        "sub_task": sub_task,
        "result": {hostname: reply_text},
        "status": "success",
    }


async def report_generate(state: AgentState, node: Node):
    """
    节点执行函数，根据当前节点类型来调用分机部署的agent
    """
    # 要发送的数据JSON-RPC 2.0规范
    agent = "report_generate"
    print(f"🤖 Agent {agent} 执行中...")
    logger.info(f"🤖 Agent {agent} 执行中...")

    hostname  = node.hostname

    report_lines = [
        f"### {host}\n```\n{output}\n```"
        for host, output in state["result"].items()
    ]
    report_lines.append(state['query'])
    full_report = "\n".join(report_lines)
    state['sub_task'] = full_report

    push_user_turn(state, hostname, full_report)

    # 调用 agent
    payload = jsonrpc_request(state, agent, node)
    result = await agent_quest(agent, node, payload)

    reply_text = result2reply_text(result)

    # 记录响应
    push_agent_turn(state, hostname, node.agent_type, reply_text)

    return {
        "hostname": hostname,
        "sub_task": full_report,
        "result": {hostname: reply_text},
        "status": "success",
    }


# agent调用索引，按照节点名称配置合适的节点执行函数
agent_dict = {"系统感知Agent": system_perception, "异常分析Agent": anomaly_analysis, "策略规划Agent": strategy_plan,
            "操作执行Agent": command_run, "报告反馈Agent": report_generate}


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
        print("异常处理节点创建完成")


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
        checkpointer = InMemorySaver()
        self.graph = workflow.compile()


    # def draw_graph(self):
    #     self.graph.get_graph().print_ascii()


    def draw_graph(self):
        output_path = "agent-a6000/01_task_analysis/graph_figs/dag_workflow.png"

        self.graph.get_graph().draw_png(
            output_file_path = output_path,
            fontname = "Noto Sans CJK SC",
        )


    async def run_workflow(self, task):
        initial_state = {
            "messages": [],
            "query": task,
            "result": {},
            "status": 'success',
            "error_code": 0,
            "error_message": '',
            "hostname": None,
            "sub_task": None,
            "expert": None
        }

        # 调用 graph，传入初始 state
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        push_user_turn(initial_state, None, task)
        await self.graph.ainvoke(initial_state, config)


app = FastAPI()

# 允许跨域请求（开发调试方便）
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = False,
    allow_methods = ["*"],
    allow_headers = ["*"],
    expose_headers = ["*"],
)


@app.get("/")
def root():
    return {"message": "This is the task analysis agent."}


@app.post("/task_analysis")
async def run(request: Request) -> Response:
    req_json = await request.json()
    task = req_json if isinstance(req_json, str) else req_json.get("params", {}).get("query", "")

    node_dict, start = generate_dag_dict(task)    # 根据任务生成 DAG

    graph = WorkGraph(node_dict, start)
    graph.dag_to_langgraph()    # 给 LangGraph 添加节点和边，完成编译
    graph.draw_graph()    # 可视化 DAG

    await graph.run_workflow(task)    # 开始执行工作流


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=5000)


if __name__ == '__main__':
    main()



# async def main() -> None:
#     task = "分析 NAVI, serve1, test_server 三台服务器的 gpu 异常原因并执行解决方案"
#     node_dict, start = generate_dag_dict(task)    # 根据任务生成 DAG

#     graph = WorkGraph(node_dict, start)    # 实例化
#     graph.dag_to_langgraph()    # 给 LangGraph 添加节点和边，完成编译
#     graph.draw_graph()    # 可视化 DAG
#     await graph.run_workflow(task)    # 开始执行工作流


# if __name__ == '__main__':
#     start_main_time = time.perf_counter()
#     asyncio.run(main())
#     end_main_time = time.perf_counter()
#     print(f'Run time: {end_main_time - start_main_time:.4f}s')
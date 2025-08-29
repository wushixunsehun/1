import os, sys, json
sys.path.append('./')
from pathlib import Path
from agentsAPI import query_llm
import socket, uvicorn, logging
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.tasks import InMemoryTaskStore
from milvus_client import PlaybookMilvusClient
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.server.agent_execution import AgentExecutor, RequestContext


log_dir = Path(__file__).resolve().parent / "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename = log_dir / "strategy_plan_agent.log",
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.propagate = False


STRATEGY_PROMPT_TEMPLATE = """你是一个自动化运维策略规划专家。你的任务是根据一台或多台服务器的异常描述，分析各自的核心异常原因，为每台服务器选择 1～5 个合适的剧本来修复异常问题，输出为“服务器名称: 可选剧本列表”的字典结构。

当前系统的诊断信息（包括执行过的命令、输出结果和部分异常分析）：

{query}

与异常描述相关的可选 ansible 剧本：

{playbook_list}

不需要解释，仅输出 JSON 格式结果，例如：
{{
    "host1": [
        {{
            "name": "...",
            "description": "...",
            "file": "...",
            "parameters": "[
                {{"name": "...", "type": "..."}},
                {{"name": "...", "type": "..."}},
                ...
            ]"
        }},
        ...
    ],
    "host2": [
        {{
            "name": "...",
            "description": "...",
            "file": "...",
            "parameters": "[
                {{"name": "...", "type": "..."}},
                {{"name": "...", "type": "..."}},
                ...
            ]"
        }},
        ...
    ]
    ...
}}

如果没有合适的剧本，请返回“无合适剧本”。
"""


async def select_playbooks(query: str, playbook_tools) -> list[dict]:
    """
    llm 推理决策执行剧本
    """
    logger.info(f"Input：\n{query}")

    playbook_list_str = '\n'.join([
        f"- {tool['name']}: {tool['description']} (文件名: {tool['file']})\n{tool['content']}" for tool in playbook_tools
    ])
    prompt = STRATEGY_PROMPT_TEMPLATE.format(query=query, playbook_list=playbook_list_str)
    response = query_llm(prompt)
    selected_playbooks = json.loads(response)
    return selected_playbooks


def build_cmd(pb_path:str, extra_vars:dict[str,str]|None) -> str:
    """
    组装 ansible 剧本执行命令
    """
    cmd = ["ansible-playbook", "-i inventory", pb_path]
    if extra_vars:
        cmd += ["-e", json.dumps(extra_vars, ensure_ascii=False)]
    return " ".join(cmd)


async def handle_payload(exception_info: str, playbook_tools) -> dict:
    try:
        data = json.loads(exception_info)
    except json.JSONDecodeError:
        data = {"query": exception_info.strip()}

    # --- 阶段 1: 任务描述 -> 候选剧本 ---
    if "query" in data and len(data) == 1:
        cands = await select_playbooks(data["query"], playbook_tools)
        return {"status": "need_user_choice", "candidates": cands}

    # --- 阶段 2: 用户选中了剧本 ID 列表 ---
    if "chosen_paths" in data:
        playbooks = []
        for playbook in data["chosen_paths"]:
            pb = next((p for p in playbook_tools if p["playbook_path"] == playbook['path']), None)
            if not pb:
                continue
            playbooks.append({
                "name": playbook['name'],
                "path": playbook['path'],
                "params": playbook['params']
            })
        return {"status": "need_params", "playbooks": playbooks}

    # --- 阶段 3: 用户提交参数值 ---
    if "param_values" in data:
        cmds = [
            build_cmd(item["path"], item.get("extra_vars"))
            for item in data["param_values"]
        ]
        logger.info(f"Generate shell commands：\n{cmds}")
        return {"status": "done", "shell_cmds": cmds}

    return {"status": "error", "msg": "payload 字段无法识别"}


# --- 8< --- 业务层 --- 8< ---
class StrategyExecutor(AgentExecutor):
    def __init__(self):
        self.m_client = PlaybookMilvusClient()
        # self.tools = load_playbook_tools(self.playbooks)


    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_input = context.get_user_input()
        # playbooks = self.m_client.list_all()    # [{'name':..., 'description':..., 'shell':...}, ...]
        playbooks = self.m_client.search_playbooks_by_name(user_input, top_k=20)
        result = await handle_payload(user_input, playbooks)
        await event_queue.enqueue_event(new_agent_text_message(json.dumps(result, ensure_ascii=False)))


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


# --- 8< --- Agent Card --- 8< ---
skill = AgentSkill(
    id = "strategy_plan",
    name = "Strategy plan",
    description = "根据系统状态数据和异常分析情况制定自愈策略",
    tags = []
)

# 获取真实主机名
hostname = socket.gethostname()
capabilities = AgentCapabilities(streaming=True)
agent_card = AgentCard(
    name = "strategy_plan",
    description = "策略规划 Agent（A2A 版）",
    url = f"http://{hostname}:5003/",
    version = "1.0.0",
    defaultInputModes = ["text/plain"],
    defaultOutputModes = ["text/plain"],
    capabilities = capabilities,
    skills = [skill],
    authentication = {"scheme": "Bearer"},
)


# --- 8< --- 组装 Starlette App  --- 8< ---
request_handler = DefaultRequestHandler(
    agent_executor = StrategyExecutor(),
    task_store = InMemoryTaskStore(),
)

server_app_builder = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)


# 自动把 Card 暴露在 /.well-known/agent.json
WELL_KNOWN_DIR = Path(__file__).resolve().parent / ".well-known"
WELL_KNOWN_DIR.mkdir(exist_ok=True)
(WELL_KNOWN_DIR / "agent.json").write_text(agent_card.model_dump_json(indent=2))


if __name__ == '__main__':
    uvicorn.run(server_app_builder.build(), host = "0.0.0.0", port = 5003)

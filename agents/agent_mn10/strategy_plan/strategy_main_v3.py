import hashlib
import os, sys, json
sys.path.append('./')
from pathlib import Path
from agentsAPI import query_llm
import socket, uvicorn, logging
from a2a.server.events import EventQueue, TaskStatusUpdateEvent
from a2a.utils import new_agent_text_message
from a2a.utils import new_agent_text_message
from milvus_client import PlaybookMilvusClient
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (AgentCard, AgentCapabilities, AgentSkill,
                        TaskStatus, TaskState, TextPart, Part
)
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


STRATEGY_PROMPT_TEMPLATE = """你是自动化运维剧本规划专家。你将收到一份诊断报告，包括一台或多台服务器的**异常描述、已执行命令及其输出、异常分析**，以及针对这些异常可用的 ansible 剧本库（包含剧本基本信息和所需参数）。

你的任务是：
1. 对每台服务器，**独立分析**其异常根因，确保聚焦于本机问题。
2. 为每台服务器**精确匹配**1~5个最合适的剧本（可少于5个，剧本越少越好，能修好为止），**只选与异常直接相关、能解决当前问题的剧本**。
3. 合理安排剧本执行顺序，**避免无关或重复修复步骤**。
4. 若无剧本可用，或无法判断有无合适剧本，请输出 "无合适剧本"。

输出格式严格如下，内容为**JSON**，无须额外解释说明：

{{
    "host1": [
        {{
            "name": "剧本名",
            "description": "剧本简介",
            "file": "剧本文件名",
            "parameters": [
                {{"name": "参数名", "type": "参数类型"}}
                // 如剧本无需传参可省略 parameters 字段
            ]
        }},
        // ...
    ],
    "host2": [
        // ...
    ]
}}

**注意：**  
- 剧本匹配必须基于异常本身、可用剧本描述，**不允许主观臆断或推测缺失信息**。
- **剧本少优于多，相关优于覆盖。**
- 如果输入的异常描述或剧本库信息不充分，允许返回 "无合适剧本"。
- **只输出标准 JSON，绝不添加注释、说明、理由或多余文字。**

当前系统的诊断信息（含已执行命令、异常输出、部分分析）：
{query}

与异常描述相关的可选 ansible 剧本列表：
{playbook_list}
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


async def handle_payload(exception_info: str, playbook_tools) -> dict:
    try:
        data = json.loads(exception_info)
    except json.JSONDecodeError:
        data = {"query": exception_info.strip()}

    if "query" in data and len(data) == 1:
        cands = await select_playbooks(data["query"], playbook_tools)
        logger.info(f"Selected playbooks: {cands}")
        tasks = []
        for host, playbooks in cands.items():
            if playbooks == "无合适剧本":
                continue
            for pb in playbooks:
                # 生成唯一 playbook 路径（host+剧本名+hash）
                pb_content = None
                for tool in playbook_tools:
                    if tool["file"] == pb["file"]:
                        pb_content = tool["content"]
                        break
                if not pb_content:
                    continue
                # 用内容 hash 保证唯一性
                content_hash = hashlib.md5(pb_content.encode("utf-8")).hexdigest()[:8]
                pb_basename = Path(pb["file"]).stem
                playbook_path = f"/tmp/ansible_playbooks/{pb_basename}_{content_hash}.yml"
                shell_cmd = f"ansible-playbook -i inventory -l {host} {playbook_path}"
                tasks.append({
                    "host": host,
                    "playbook_path": playbook_path,
                    "playbook_content": pb_content,
                    "shell_cmd": shell_cmd
                })
        if not tasks:
            return {"status": "success", "tasks": [], "msg": "无合适剧本"}
        return {"status": "success", "tasks": tasks}

    return {"status": "error", "msg": "payload 字段无法识别"}


# --- 8< --- 业务层 --- 8< ---
class StrategyExecutor(AgentExecutor):
    def __init__(self):
        self.m_client = PlaybookMilvusClient()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_input = context.get_user_input()
        playbooks = self.m_client.search_playbooks_by_name(user_input, top_k=20)
        # 采用流式 LLM 推理
        async for token in query_llm(
            STRATEGY_PROMPT_TEMPLATE.format(query=user_input, playbook_list='\n'.join([
                f"- {tool['name']}: {tool['description']} (文件名: {tool['file']})\n{tool['content']}" for tool in playbooks
            ])),
            stream=True
        ):
            # 检查特殊交互事件
            if token.startswith("__CHOICE__"):
                # 解析choice内容，推送choice事件
                # 例如: __CHOICE__{"options": [...], "prompt": "请选择剧本"}
                try:
                    event_data = json.loads(token[len("__CHOICE__"):])
                    await event_queue.enqueue_event(new_agent_choice_event(event_data))
                    # 等待用户输入
                    user_choice = await event_queue.wait_for_user_response()
                    await event_queue.enqueue_event(new_agent_text_message(f"用户选择: {user_choice}"))
                except Exception as e:
                    await event_queue.enqueue_event(new_agent_text_message(f"[交互事件解析失败]{e}"))
                continue
            elif token.startswith("__INPUT__"):
                # 解析input内容，推送input事件
                try:
                    event_data = json.loads(token[len("__INPUT__"):])
                    await event_queue.enqueue_event(new_agent_input_event(event_data))
                    user_input_val = await event_queue.wait_for_user_response()
                    await event_queue.enqueue_event(new_agent_text_message(f"用户输入: {user_input_val}"))
                except Exception as e:
                    await event_queue.enqueue_event(new_agent_text_message(f"[交互事件解析失败]{e}"))
                continue
            # 普通token直接推送
            await event_queue.enqueue_event(new_agent_text_message(token))

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

import os
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('.../')


import json
import httpx
import uvicorn
from pathlib import Path
from agent_state import AgentState
from agent_mn10.system_perception.experts_run_v2 import build_graph
from fastapi.staticfiles import StaticFiles

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier


# ---------- 1. 业务层 ----------
class PerceptionExecutor(AgentExecutor):
    def __init__(self):
        self.graph = build_graph()


    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # 兼容二次确认参数
        user_input = context.get_user_input()
        confirm = None
        cmds_for_confirm = None
        query = user_input
        # 支持 json 格式输入
        try:
            if isinstance(user_input, str) and user_input.strip().startswith('{'):
                user_input_json = json.loads(user_input)
                query = user_input_json.get("query", "")
                confirm = user_input_json.get("confirm")
                cmds_for_confirm = user_input_json.get("cmds_for_confirm")
        except Exception:
            pass
        expert = "state"

        initial_state: AgentState = {
            "messages": [],    # 用空列表
            "query": query,
            "sub_task": query,    # 可选：与 query 同值或自行拆分
            "expert": expert,
            "hostname": None,
            "result": {},
            "status": "success",    # 默认成功
            "error_code": 0,
            "error_message": "",
            "confirm": confirm,
            "cmds_for_confirm": cmds_for_confirm,
        }

        result = await self.graph.ainvoke(initial_state)
        payload = result["result"]
        
        # 适配 need_confirm/cancelled 直接透传
        if isinstance(payload, dict) and payload.get("status") in ("need_confirm", "cancelled"):
            # 需要把 cmds_for_confirm 也带到响应里
            payload_out = dict(payload)
            if "cmds_for_confirm" not in payload_out and "cmds_for_confirm" in result:
                payload_out["cmds_for_confirm"] = result["cmds_for_confirm"]
            txt = json.dumps(payload_out, ensure_ascii=False, indent=2)
        else:
            try:
                txt = json.dumps(payload['data'], ensure_ascii=False, indent=2)
            except TypeError:
                txt = str(payload)

        # 流式输出结果
        chunk_size = 100  # 每次发送的字符数
        for i in range(0, len(txt), chunk_size):
            chunk = txt[i:i+chunk_size]
            await event_queue.enqueue_event(new_agent_text_message(chunk))


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


# ---------- 2. Agent Card ----------
skill = AgentSkill(
    id = "system_perception",
    name = "System perception",
    description = "通过 shell 命令获取系统状态",
    tags = []
)

capabilities = AgentCapabilities(streaming=True)
agent_card = AgentCard(
    name = "system_perception",
    description = "系统感知 Agent（A2A 版）",
    url = "http://localhost:5001/",
    version = "1.0.0",
    defaultInputModes = ["text/plain"],
    defaultOutputModes = ["text/plain"],
    capabilities = capabilities,
    skills = [skill],
)


# ---------- 3. 组装 Starlette App ----------
request_handler = DefaultRequestHandler(
    agent_executor = PerceptionExecutor(),
    task_store = InMemoryTaskStore(),
)

server_app_builder = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)


# 自动把 Card 暴露在 /.well-known/agent.json
WELL_KNOWN_DIR = Path(__file__).resolve().parent / ".well-known"
WELL_KNOWN_DIR.mkdir(exist_ok=True)
(WELL_KNOWN_DIR / "agent.json").write_text(agent_card.model_dump_json(indent=2))


def main() -> None:
    uvicorn.run(server_app_builder.build(), host = "0.0.0.0", port = 5001)


if __name__ == '__main__':
    main()

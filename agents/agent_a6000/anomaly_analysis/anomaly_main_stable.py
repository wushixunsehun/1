import os, sys, json
sys.path.append('./')

import socket
import uvicorn
from pathlib import Path
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier


# ---------- 1. 业务层 ----------
class AnomalyAnalysisExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        
        # 调用系统图
        result = await self.system_graph.ainvoke({"query": query})
        response_data = result.get("response", {})
        
        # 转换为文本并流式输出
        txt = json.dumps(response_data, ensure_ascii=False, indent=2)
        
        # 流式输出结果
        chunk_size = 100  # 每次发送的字符数
        for i in range(0, len(txt), chunk_size):
            chunk = txt[i:i+chunk_size]
            await event_queue.enqueue_event(new_agent_text_message(chunk))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


# ---------- 2. Agent Card ----------
skill = AgentSkill(
    id = "anomaly_analysis",
    name = "Anomaly Analysis",
    description = "异常分析和根因定位",
    tags = []
)

# 获取真实主机名
hostname = socket.gethostname()
capabilities = AgentCapabilities(streaming=True)
agent_card = AgentCard(
    name = "anomaly_analysis",
    description = "异常分析 Agent（A2A 版）",
    url = f"http://{hostname}:5002/",
    version = "1.0.0",
    defaultInputModes = ["text/plain"],
    defaultOutputModes = ["text/plain"],
    capabilities = capabilities,
    skills = [skill],
    authentication = {"scheme": "Bearer"},
)


# ---------- 3. 组装 Starlette App ----------
request_handler = DefaultRequestHandler(
    agent_executor = AnomalyAnalysisExecutor(),
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
    # 启动 A2A 服务器（支持流式）
    uvicorn.run(server_app_builder.build(), host = "0.0.0.0", port = 5002)

if __name__ == '__main__':
    main()


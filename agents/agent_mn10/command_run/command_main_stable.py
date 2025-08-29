import os
import sys
sys.path.append('./')
from pathlib import Path

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent


base_dir = get_base_dir()
log_dir = base_dir / 'logs'
log_dir.mkdir(exist_ok=True)


import socket
import uvicorn
import logging
from pathlib import Path
from langchain_community.tools import ShellTool


from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier


logging.basicConfig(
    filename = os.path.join(log_dir, "command_run_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_command(commands):
    """
    执行系统感知传来的 shell 命令字符串（已使用 && 连接多个命令）
    """
    if not isinstance(commands, str):
        return "[Error] Input must be a single string of shell commands."

    shell_tool = ShellTool()

    try:
        output = await shell_tool.arun(commands)
        return f"$ {commands}\n{output}"
    except Exception as e:
        return f"[Error executing command]: {e}"


class CommandExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        logger.info(f"输入命令：{query}")

        result_text = await run_command(query)
        logger.info(f"命令执行输出：\n{result_text}")

        await event_queue.enqueue_event(new_agent_text_message(result_text))


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


# ---------- 2. Agent Card ----------
skill = AgentSkill(
    id = "command_run",
    name = "Command run",
    description = "在服务器上执行一些命令操作",
    tags = []
)

# 获取真实主机名
hostname = socket.gethostname()
capabilities = AgentCapabilities(streaming=True)
agent_card = AgentCard(
    name = "command_run",
    description = "操作执行 Agent（A2A 版）",
    url = f"http://{hostname}:5004/",
    version = "1.0.0",
    defaultInputModes = ["text/plain"],
    defaultOutputModes = ["text/plain"],
    capabilities = capabilities,
    skills = [skill],
    authentication = {"scheme": "Bearer"},
)


# ---------- 3. 组装 Starlette App ----------
request_handler = DefaultRequestHandler(
    agent_executor = CommandExecutor(),
    task_store = InMemoryTaskStore(),
)

server_app_builder = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)


# 自动把 Card 暴露在 /.well-known/agent.json
WELL_KNOWN_DIR = base_dir / '.well-known'
WELL_KNOWN_DIR.mkdir(exist_ok=True)
(WELL_KNOWN_DIR / "agent.json").write_text(agent_card.model_dump_json(indent=2))


def main() -> None:
    uvicorn.run(server_app_builder.build(), host = "0.0.0.0", port = 5004)


if __name__ == '__main__':
    main()

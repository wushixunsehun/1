import sys
sys.path.append('./')


import json
import socket
import uvicorn
import uuid
import time
import yaml
import logging
import os
import re
from pathlib import Path
from typing import AsyncIterator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from agent_state import AgentState
from agent_mn10.system_perception.experts_run_stable import build_graph, state_expert_stream


from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier


# 配置日志
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename = os.path.join(log_dir, "system_perception_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 加载配置
agents_dir = Path(__file__).resolve().parents[2]
config_path = agents_dir / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

async def stream_json_output(raw_stream: AsyncIterator[dict]) -> AsyncIterator[str]:
    """
    流式处理系统感知输出，格式化为易读的文本
    """
    async for data in raw_stream:
        status = data.get("status")
        
        if status == "progress":
            message = data.get("message", "")
            
            # 处理不同类型的进度信息
            if "commands" in data:
                commands = data["commands"]
                yield f"{message}\n\n"
                for host, cmds in commands.items():
                    yield f"【{host}】: `{', '.join(cmds)}`\n\n"
            elif "host_result" in data:
                host_result = data["host_result"]
                for host, output in host_result.items():
                    # yield f"【{host}】执行结果\n\n"
                    yield f"```bash\n{output}\n```\n\n"
            else:
                pass
                # yield f"**状态**: {message}\n\n"
                
        elif status == "success":
            result = data.get("result", {})
            if isinstance(result, dict) and "output" in result:
                # 格式化最终命令输出
                output = result["output"]
                # yield f"## ✅ 系统状态信息汇总\n\n{output}\n\n"
            else:
                yield f"**结果**: {json.dumps(result, ensure_ascii=False, indent=2)}\n\n"
                
        elif status == "error":
            error_msg = data.get("error_message", "未知错误")
            yield f"## ❌ 错误\n\n{error_msg}\n\n"
            
        else:
            # 其他状态信息
            # pass
            yield f"**状态**: {json.dumps(data, ensure_ascii=False, indent=2)}\n\n"


async def generate_system_perception_stream(query: str) -> AsyncIterator[dict]:
    """
    生成系统感知的流式输出
    """
    try:
        expert = "state"  # 默认使用 state expert

        initial_state: AgentState = {
            "messages": [],
            "query": query,
            "sub_task": query,
            "expert": expert,
            "hostname": None,
            "result": {},
            "status": "success",
            "error_code": 0,
            "error_message": "",
        }

        # 直接使用流式 expert 函数
        async for result in state_expert_stream(initial_state):
            # 确保 result 结构正确
            if "result" in result and isinstance(result["result"], dict):
                if "output" not in result["result"] and result.get("status") == "success":
                    result["result"] = {"output": result["result"]}
            
            yield result

    except Exception as e:
        logger.error(f"系统感知流式处理错误: {str(e)}")
        yield {
            "status": "error",
            "error_message": str(e),
            "result": {}
        }


# ---------- 1. 业务层 ----------
class PerceptionExecutor(AgentExecutor):
    def __init__(self):
        self.graph = build_graph()


    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        
        # 使用流式生成器
        raw_stream = generate_system_perception_stream(query)
        shaped_stream = stream_json_output(raw_stream)
        
        async for token in shaped_stream:
            await event_queue.enqueue_event(new_agent_text_message(token))


    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


# ---------- 2. Agent Card ----------
skill = AgentSkill(
    id = "system_perception",
    name = "System perception",
    description = "通过 shell 命令获取系统状态",
    tags = []
)

# 获取真实主机名
hostname = socket.gethostname()
capabilities = AgentCapabilities(streaming=True)
agent_card = AgentCard(
    name = "system_perception",
    description = "系统感知 Agent（A2A 版）",
    url = f"http://{hostname}:5001/",
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


# ---------- 4. HTTP流式接口 ----------
http_app = FastAPI()

@http_app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    stream = data.get("stream", False)
    
    # 提取查询内容
    if "messages" in data and isinstance(data["messages"], list):
        query = data["messages"][-1].get("content", "")
    else:
        query = data.get("query", "")
    
    if not query:
        return JSONResponse({"error": "No query provided"}, status_code=400)
    
    if stream:
        # 真实流式输出
        async def generate_stream():
            try:
                raw_stream = generate_system_perception_stream(query)
                shaped_stream = stream_json_output(raw_stream)

                async for token in shaped_stream:
                    if token:
                        chunk = {
                            "id": str(uuid.uuid4()),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "system-perception-agent",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Stream generation error: {str(e)}")
                error_chunk = {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "system-perception-agent",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"Error: {str(e)}"},
                        "finish_reason": "stop"
                    }]
                }
                try:
                    yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                    # 客户端已断开，忽略错误
                    pass
                
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    else:
        # 非流式输出
        try:
            raw_stream = generate_system_perception_stream(query)
            result_data = {}
            async for data in raw_stream:
                result_data = data
            
            return JSONResponse({
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "system-perception-agent",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(result_data, ensure_ascii=False)},
                    "finish_reason": "stop"
                }]
            })
        except Exception as e:
            logger.error(f"Non-stream generation error: {str(e)}")
            return JSONResponse({"error": str(e)}, status_code=500)


# ---------- 5. 合并应用程序 ----------
# 创建合并的应用
combined_app = FastAPI()
combined_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载HTTP流式接口
@combined_app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: Request):
    return await chat_completions(request)

# 挂载A2A服务器
a2a_app = server_app_builder.build()
combined_app.mount("/a2a", a2a_app)

# 重新导出A2A端点到根路径
@combined_app.get("/.well-known/agent.json")
async def get_agent_card():
    return agent_card.model_dump()


def main() -> None:
    # 启动合并的服务器（同时支持A2A和HTTP流式）
    uvicorn.run(combined_app, host="0.0.0.0", port=5001)


if __name__ == '__main__':
    main()

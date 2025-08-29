import os, re, sys
sys.path.append('./')
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)


import yaml
import logging, json
import socket, uvicorn
import time, uuid
from pathlib import Path
from typing import Tuple, AsyncIterator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from agentsAPI import query_llm, get_rag_rpc, strip_think
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier


agents_dir = Path(__file__).resolve().parents[2]
config_path = agents_dir / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


logging.basicConfig(
    filename = os.path.join(log_dir, "report_generate_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.propagate = False


QA_PROMPT = """你是一名领域内资深技术顾问，擅长将复杂的专业问题转化为清晰、权威的知识解答。

你的任务是结合【上下文】与【用户问题】，输出一份**深度专业、条理清晰**的答复，以支持用户高效决策与理解。

### 解答要求：
1. **结合上下文优先答题**，优先引用上下文中的事实、命令输出、描述性线索来作答。
2. **如上下文不足**，可适当补充背景知识，但应保持与用户问题的紧密相关性。
3. 答案应体现领域知识深度，避免空洞、表面化回答。
4. **输出格式严格使用标准 Markdown 格式**
5. 确保换行结构清晰，便于前端渲染。
6. 直接输出内容，不要使用```markdown包裹。

### 格式要求：
- 标题前后必须有空行（如：\n# 标题 \n）
- 标题符号之后需要有一个空格（如：# 标题）
- 列表项之间无需空行，但列表前后要有空行
- 代码块前后必须有空行（如：\n```bash\n ... \n```\n）
- 表格前后必须有空行
- 水平线前后必须有空行

【上下文】
{context}

【用户问题】
{query}
"""


HISTORY_PROMPT = """你是一名资深系统运维分析师，擅长从命令输出与运维对话中提炼问题本质，并撰写清晰、专业、可操作的系统分析报告。

你的任务是结合【对话记录/上游命令输出】与【当前任务】，输出一份结构清晰、重点突出的分析报告。

### 报告应围绕以下方向展开：
- 当前的故障现象或关键异常（包括命令输出中的报错、指标异常、状态波动等）
- 可能原因及推理过程（结合上下文做判断，若有多个可能性可适当排序）
- 操作建议或应对措施（具体、可执行，避免泛泛而谈）
- 总结与后续建议（强调处理重点，指出是否还需持续关注或进一步排查）

### 写作要求：
- 报告结构不强制固定，可自由组织标题，风格应真实、专业、有现场感
- **输出格式严格使用标准 Markdown 格式**
- 确保换行结构清晰，便于前端渲染
- 直接输出内容，不要使用```markdown包裹

### 格式要求：
- 标题前后必须有空行（如：\n# 标题 \n）
- 标题符号之后需要有一个空格（如：# 标题）
- 列表项之间无需空行，但列表前后要有空行
- 代码块前后必须有空行（如：\n```bash\n ... \n```\n）
- 表格前后必须有空行
- 水平线前后必须有空行

【对话记录/上游命令输出】
{history}

【当前任务】
{query}
"""


HOST_BLOCK_RE = re.compile(r"###\s*(?P<host>[^\n]+)\n```(?P<block>.*?)```",re.S | re.I,)


async def stream_md_charwise(raw_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    简单的字符流式处理，只处理 think 标签，其他直接输出
    """
    in_think_block = False
    
    async for chunk in raw_stream:
        # 处理 <think> 标签的特殊情况
        if in_think_block:
            if '</think>' in chunk:
                before_end, after_end = chunk.split('</think>', 1)
                yield before_end
                yield '</think>\n\n'
                in_think_block = False
                # 继续处理剩余内容
                if after_end:
                    chunk = after_end
                else:
                    continue
            else:
                # 仍在 think 块内，直接输出
                yield chunk
                continue
        
        # 检查新的 <think> 开始
        if not in_think_block and '<think>' in chunk:
            before_start, after_start = chunk.split('<think>', 1)
            
            # 输出 <think> 前的内容
            if before_start:
                yield before_start
            
            yield '<think>\n'
            in_think_block = True
            
            # 检查是否在同一 chunk 中结束
            if '</think>' in after_start:
                think_content, remaining = after_start.split('</think>', 1)
                yield think_content
                yield '</think>\n\n'
                in_think_block = False
                
                # 处理剩余内容
                if remaining:
                    yield remaining
            else:
                # think 块未结束，输出剩余内容
                if after_start:
                    yield after_start
            continue
        
        # 普通内容直接输出
        yield chunk
    
    # 确保内容结束时有换行
    yield '\n'


def split_report(full_report: str) -> Tuple[str, str]:
    """
    拆分 agent 的 full_report:
    1. history: 所有 '## Agent: xxx' + '### host' 片段（合并为一个字符串）
    2. query_rest: 去掉上述片段后剩下的文本（如任务描述等）
    """
    history_parts = []
    rest = full_report

    # 匹配所有 '##\s*Agent:[^\n]+' 行
    agent_matches = list(re.finditer(r"^##\s*Agent:[^\n]+", full_report, flags=re.M))

    # 匹配所有 '### host' 片段
    host_matches = list(HOST_BLOCK_RE.finditer(full_report))

    # 逐个组装 history，每个片段前加 agent header（如有）
    for _, m in enumerate(host_matches):
        host = m.group("host").strip()
        block = m.group("block").strip()
        if host.lower() != "none":
            # 取最近的 agent header（如有）
            agent_header = ""
            for a in reversed(agent_matches):
                if a.start() < m.start():
                    agent_header = a.group(0).strip()
                    break
            if agent_header:
                history_parts.append(f"{agent_header}\n### {host}\n```\n{block}\n```")
            else:
                history_parts.append(f"### {host}\n```\n{block}\n```")
        rest = rest.replace(m.group(0), "")

    # 处理完所有 host 片段后，再去掉所有 agent header
    rest = re.sub(r"^##\s*Agent:[^\n]+\n?", "", rest, flags=re.M)

    history = "\n".join(history_parts).strip()
    query_rest = rest.strip()

    return history, query_rest


# --- 替换原有 generate_report 为异步生成器 ---
async def generate_report_stream(query: str, source: list, need_rag: bool):
    history, query_simple = split_report(query)
    context = get_rag_rpc(query, source) if need_rag else ""
    if history:
        logger.info(f"对话记录：\n{history}\n输入查询：\n{query_simple}")
        prompt = HISTORY_PROMPT.format(
            history = history,
            query = query_simple,
        )
    else:
        logger.info(f"输入查询：\n{query_simple}")
        prompt = QA_PROMPT.format(
            context = context,
            query = query_simple,
        )
    enable_thinking = config.get("report_think")

    async for token in query_llm(prompt, stream=True, enable_thinking=enable_thinking):
        yield token


# ---------- 1. 业务层 ----------
class ReportExecutor(AgentExecutor):
    def __init__(self):
        self.source = ['system', 'lustre', 'promql', 'slurm', 'handbook', 'ticket', 'general']
        self.need_rag = config.get("report_need_rag")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        raw_stream = generate_report_stream(query, self.source, self.need_rag)
        shaped = stream_md_charwise(raw_stream)
        async for token in shaped:
            await event_queue.enqueue_event(new_agent_text_message(token))

            
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


# ---------- 2. Agent Card ----------
skill = AgentSkill(
    id = "report_generate",
    name = "Report generate",
    description = "根据对话记录和上下文生成报告",
    tags = []
)

# 获取真实主机名
hostname = socket.gethostname()
capabilities = AgentCapabilities(streaming=True)
agent_card = AgentCard(
    name = "report_generate",
    description = "报告反馈 Agent（A2A 版）",
    url = f"http://{hostname}:5105/",
    version = "1.0.0",
    defaultInputModes = ["text/plain"],
    defaultOutputModes = ["text/plain"],
    capabilities = capabilities,
    skills = [skill],
    authentication = {"scheme": "Bearer"},
)


# ---------- 3. 组装 Starlette App ----------
request_handler = DefaultRequestHandler(
    agent_executor = ReportExecutor(),
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
                raw_stream = generate_report_stream(
                    query,
                    ['system', 'lustre', 'promql', 'slurm', 'handbook', 'ticket', 'general'],
                    config.get("report_need_rag")
                )

                async for token in stream_md_charwise(raw_stream):
                    if token:
                        chunk = {
                            "id": str(uuid.uuid4()),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "report-agent",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None
                            }]
                        }
                        try:
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                            logger.warning("Client disconnected during streaming")
                            break
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Stream generation error: {str(e)}")
                error_chunk = {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "report-agent",
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


# ---------- 5. 合并应用程序 ----------
from fastapi.middleware.cors import CORSMiddleware

# 添加CORS支持
http_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 将A2A服务器挂载到HTTP服务器
from fastapi import FastAPI
from starlette.applications import Starlette
from starlette.routing import Mount

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

@combined_app.post("/v1/agent/messages")
async def a2a_messages(request: Request):
    # 转发到A2A应用
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:5105/a2a/v1/agent/messages",
            content=await request.body(),
            headers=dict(request.headers)
        )
        return response.json()


def main() -> None:
    # 启动合并的服务器（同时支持A2A和HTTP流式）
    uvicorn.run(combined_app, host="0.0.0.0", port=5105)


if __name__ == '__main__':
    main()

import os, re, sys
sys.path.append('./')
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)


import yaml, httpx
import logging, json
import socket, uvicorn
import time, uuid
from pathlib import Path
from fastapi import FastAPI, Request
from typing import Tuple, AsyncIterator
from a2a.server.events import EventQueue
from fastapi.responses import JSONResponse, StreamingResponse
from a2a.utils import new_agent_text_message
from fastapi.middleware.cors import CORSMiddleware
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
4. **输出格式严格要求**：
   - 使用标准Markdown格式
   - 所有标题后必须有空格：## 标题，## 2. 标题（不是##标题）
   - 所有列表项后必须有空格：1. 项目，- 项目（不是1.项目）
   - 标题之后要有空行（## ... \n）
   - 段落之间要有空行分隔
   - 表格前后要有空行
   - 代码块前后要有空行（\n```bash\n ... ```\n）
5. 确保换行结构清晰，便于前端渲染。
6. 直接输出内容，不要使用```markdown包裹。
7. 可以适当添加符合情境的 emoji 表情来增强可读性，但不要过度使用。
8. 无需表明报告的生成时间和报告人

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
- **输出格式严格要求**：
  - 使用标准Markdown格式
  - 所有标题后必须有空格：## 标题，## 2. 标题（不是##标题）
  - 所有列表项后必须有空格：1. 项目，- 项目（不是1.项目）
  - 标题之后要有空行（## ... \n）
  - 段落之间要有空行分隔
  - 表格前后要有空行
  - 代码块前后要有空行（\n```bash\n ... ```\n）
- 确保换行结构清晰，便于前端渲染
- 直接输出内容，不要使用```markdown包裹
- 可以适当添加符合情境的 emoji 表情来增强可读性，但不要过度使用
- 无需表明报告的生成时间和报告人

【对话记录/上游命令输出】
{history}

【当前任务】
{query}
"""


HOST_BLOCK_RE = re.compile(
    r"【(?P<host>[^\]】]+)】[：:][^\n]*\n*\n*```(?:bash)?\n(?P<block>.*?)```",
    re.S | re.I,
)


async def stream_md_charwise(raw_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    流式处理 Markdown 格式，确保正确的标题空格、列表格式和代码块
    """
    line_buffer = ""
    in_code_block = False
    previous_line_type = "start"  # start, empty, text, header, list, code, table
    pending_header_newline = False  # 标记是否需要在标题后添加空行
    
    async for chunk in raw_stream:
        for char in chunk:
            if char == '\r':
                continue
                
            if char == '\n':
                # 处理完整行
                line = line_buffer.strip()
                current_line_type = "empty" if not line else "text"
                
                if in_code_block:
                    # 代码块内，直接输出原始行
                    if line == "```":
                        yield line_buffer + "\n\n"  # 代码块结束后加空行
                        in_code_block = False
                        current_line_type = "code"
                        pending_header_newline = False
                    else:
                        yield line_buffer + "\n"
                        current_line_type = "code"
                else:
                    # 先处理待定的标题后空行
                    if pending_header_newline and line:  # 标题后有内容
                        yield "\n"  # 添加标题后的空行
                        pending_header_newline = False
                    
                    # 检查各种 Markdown 元素
                    if line.startswith("```"):
                        # 代码块开始
                        if previous_line_type not in ["empty", "header", "start"]:
                            yield "\n"  # 代码块前加空行
                        yield line_buffer + "\n"
                        in_code_block = True
                        current_line_type = "code"
                        pending_header_newline = False
                    elif re.match(r'^#{1,6}(\s|$)', line):
                        # 标题：确保 # 后有空格
                        fixed_line = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', line_buffer)
                        if previous_line_type == "text":
                            yield "\n"  # 标题前加空行（如果前面是文本）
                        yield fixed_line + "\n"
                        current_line_type = "header"
                        pending_header_newline = True  # 标记需要在标题后添加空行
                    elif re.match(r'^\s*-{3,}\s*$', line):
                        # 水平线
                        if previous_line_type not in ["empty", "start"]:
                            yield "\n"
                        yield "---\n\n"
                        current_line_type = "empty"
                        pending_header_newline = False
                    elif re.match(r'^\d+\.(\s|$)', line_buffer):
                        # 有序列表：确保 . 后有空格
                        fixed_line = re.sub(r'^(\d+\.)([^\s])', r'\1 \2', line_buffer)
                        if previous_line_type == "text":
                            yield "\n"  # 列表前加空行（如果前面是文本）
                        yield fixed_line + "\n"
                        current_line_type = "list"
                        pending_header_newline = False
                    elif re.match(r'^[-*+](\s|$)', line_buffer):
                        # 无序列表：确保标记后有空格
                        fixed_line = re.sub(r'^([-*+])([^\s])', r'\1 \2', line_buffer)
                        if previous_line_type == "text":
                            yield "\n"  # 列表前加空行（如果前面是文本）
                        yield fixed_line + "\n"
                        current_line_type = "list"
                        pending_header_newline = False
                    elif line.startswith('|') and line.endswith('|'):
                        # 表格
                        if previous_line_type not in ["empty", "table", "start"]:
                            yield "\n"
                        yield line_buffer + "\n"
                        current_line_type = "table"
                        pending_header_newline = False
                    elif line:
                        # 普通文本行 - 这里是关键修复点
                        yield line_buffer + "\n"
                        current_line_type = "text"
                        pending_header_newline = False
                    else:
                        # 空行
                        yield line_buffer + "\n"
                        current_line_type = "empty"
                        pending_header_newline = False
                
                # 更新状态
                line_buffer = ""
                previous_line_type = current_line_type
            else:
                # 累积字符到行缓冲区
                line_buffer += char
    
    # 处理最后一行（如果没有换行符结尾）
    if line_buffer:
        line = line_buffer.strip()
        
        # 处理待定的标题后空行
        if pending_header_newline and line:
            yield "\n"
            pending_header_newline = False
            
        if in_code_block:
            yield line_buffer + "\n"
        elif re.match(r'^#{1,6}(\s|$)', line):
            fixed_line = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', line_buffer)
            if previous_line_type == "text":
                yield "\n"
            yield fixed_line + "\n\n"
        elif re.match(r'^\d+\.(\s|$)', line_buffer):
            fixed_line = re.sub(r'^(\d+\.)([^\s])', r'\1 \2', line_buffer)
            if previous_line_type == "text":
                yield "\n"
            yield fixed_line + "\n"
        elif re.match(r'^[-*+](\s|$)', line_buffer):
            fixed_line = re.sub(r'^([-*+])([^\s])', r'\1 \2', line_buffer)
            if previous_line_type == "text":
                yield "\n"
            yield fixed_line + "\n"
        elif line:
            yield line_buffer + "\n"
        else:
            yield line_buffer + "\n"


def split_report(full_report: str) -> Tuple[str, str]:
    """
    拆分 agent 的 full_report:
    1. history: 所有 '## Agent: xxx' + '### host' 片段（合并为一个字符串）
    2. query_rest: 去掉上述片段后剩下的文本（如任务描述等）
    """
    # 查找所有 '## Query' 的位置（兼容冒号/空格/中英文冒号）
    matches = list(re.finditer(r"^##\s*Query[：: ]", full_report, flags=re.M))
    if matches:
        m = matches[-1]
        split_idx = m.start()
        history = full_report[:split_idx].strip()
        query_rest = full_report[split_idx:].strip()
    else:
        history = full_report.strip()
        query_rest = ""
    return history, query_rest


# --- 替换原有 generate_report 为异步生成器 ---
async def generate_report_stream(query: str, source: list, need_rag: bool):
    history, query_simple = split_report(query)
    if history:
        logger.info(f"对话记录：\n{history}\n\n输入查询：\n{query_simple}")
        prompt = HISTORY_PROMPT.format(
            history = history,
            query = query_simple,
        )
    else:
        logger.info(f"输入查询：\n{query_simple}")
        context = get_rag_rpc(query, source) if need_rag else ""
        prompt = QA_PROMPT.format(
            context = context,
            query = query_simple,
        )
    enable_thinking = config.get("report_think")
    show_think_block = config.get("report_show_think_chunk")

    if show_think_block:
        # 直接流式输出
        async for token in query_llm(
                prompt,
                stream = True,
                enable_thinking = enable_thinking,
                temperature = 0.2,
                top_p = 0.7,
                presence_penalty = 0.2,
                frequency_penalty = 0.65
            ):
            yield token
    else:
        in_think = False
        tag_buf = ""
        end_buf = ""
        async for token in query_llm(
                prompt,
                stream = True,
                enable_thinking = enable_thinking,
                temperature = 0.2,
                top_p = 0.7,
                presence_penalty = 0.2,
                frequency_penalty = 0.65
            ):
            for c in token:
                if not in_think:
                    tag_buf += c
                    while tag_buf and not "<think>".startswith(tag_buf):
                        yield tag_buf[0]
                        tag_buf = tag_buf[1:]
                    if tag_buf == "<think>":
                        tag_buf = ""
                        in_think = True
                else:
                    end_buf += c
                    if len(end_buf) > 8:
                        end_buf = end_buf[-8:]
                    if end_buf.endswith("</think>"):
                        end_buf = ""
                        tag_buf = ""
                        in_think = False
        if not in_think and tag_buf:
            yield tag_buf


# ---------- 1. 业务层 ----------
class ReportExecutor(AgentExecutor):
    def __init__(self):
        self.source = config.get("report_source")
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
    url = f"http://{hostname}:5005/",
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
                    config.get("report_source"),
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


# 添加CORS支持
http_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:5005/a2a/v1/agent/messages",
            content=await request.body(),
            headers=dict(request.headers)
        )
        return response.json()


def main() -> None:
    # 启动合并的服务器（同时支持A2A和HTTP流式）
    uvicorn.run(combined_app, host="0.0.0.0", port=5005)


if __name__ == '__main__':
    main()

import os, sys, json, re
import yaml, httpx, uuid
sys.path.append('./')
from pathlib import Path
from typing import AsyncIterator
from fastapi import FastAPI, Request
import socket, uvicorn, logging, time
from a2a.server.events import EventQueue
from agentsAPI import query_llm, strip_think
from a2a.utils import new_agent_text_message
from a2a.server.tasks import InMemoryTaskStore
from milvus_client import PlaybookMilvusClient
from a2a.client import A2ACardResolver, A2AClient
from fastapi.middleware.cors import CORSMiddleware
from a2a.server.apps import A2AStarletteApplication
from fastapi.responses import JSONResponse, StreamingResponse
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    AgentCapabilities,
    AgentSkill
)


agents_dir = Path(__file__).resolve().parents[2]
config_path = agents_dir / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


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


STRATEGY_PROMPT_TEMPLATE_v2 = """你是自动化运维剧本规划专家。你将收到一份诊断报告，包括一台或多台服务器的**异常描述、已执行命令及其输出、异常分析**，以及数据库中可用的可执行剧本/脚本（包含基本信息和所需参数）。

你的任务分三步：
1. **意图识别**：先分析诊断信息，仅识别诊断信息中提及的需要处理的服务器、任务类型、异常现象和修复目标。对每台服务器，**独立分析**其异常根因，确保聚焦于服务器问题，确保能区分多服务器/多任务场景，按服务器/任务分别处理。
2. **剧本筛选**：针对每个服务器/任务，独立判断是否有合适的剧本。**精确匹配**与异常直接相关、能解决当前问题的1~5个剧本，能修好为止。合理安排剧本执行顺序，**避免无关或重复修复步骤**。
3. **参数推理**：对每个选中的剧本，结合剧本参数定义和诊断信息，自动推理并填写所有必需参数。你需要仔细分析剧本参数的含义和用途，从服务器状态、异常信息、已执行命令上下文中提取相关值，按照参数类型给出合理的值。

若无剧本可用，或无法判断有无合适剧本，请输出 "无合适剧本"。

输出格式严格如下，内容为**JSON**，不包含任何注释、说明、理由或多余文字：

{{
    "host1": {{
        "playbooks": [
            {{
                "name": "剧本名",
                "description": "剧本简介",
                "file": "剧本文件名",
                "parameters": [
                    {{
                        "name": "参数名",
                        "type": "参数类型",
                        "value": "推理出的参数值"
                    }},
                    ...
                ]
            }}
        ],
        ...
    }},
    "host2": "无合适剧本"
}}

**注意：**
- 必须先做意图识别，分清需要处理的服务器/任务（注意：剧本里的 "hosts" 不属于要处理的主机），再筛选合适的剧本、推理参数。
- 剧本的选择必须基于异常描述本身，**不允许主观臆断或推测缺失信息**。
- 无需考虑剧本中本身的 "hosts" 字段，**只需关注剧本的功能和参数**。
- **剧本少优于多，相关优于覆盖**。
- **只输出标准 JSON，绝不添加注释、说明、理由或多余文字**。

当前系统的诊断信息（含已执行命令、异常输出、部分分析）：
{query}

与异常描述相关的可选 ansible 剧本列表：
{playbook_list}
"""


async def stream_strategy_output(raw_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    流式处理策略规划输出，确保正确的格式化
    """
    buffer = ""
    
    async for chunk in raw_stream:
        buffer += chunk
        
        # 检查是否有完整的行可以输出
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            yield line + '\n'
    
    # 输出剩余内容
    if buffer:
        yield buffer


def extract_playbook_hosts(playbook_content: str) -> str:
    """
    从playbook内容中提取hosts字段的值
    
    Args:
        playbook_content: playbook的YAML内容
        
    Returns:
        str: hosts字段的值，如果解析失败返回'unknown'
    """
    try:
        # 尝试解析YAML内容
        data = yaml.safe_load(playbook_content)
        
        # 如果是列表，取第一个play
        if isinstance(data, list) and len(data) > 0:
            play = data[0]
        elif isinstance(data, dict):
            play = data
        else:
            return 'unknown'
        
        # 提取hosts字段
        hosts = play.get('hosts', 'unknown')
        return str(hosts)
        
    except (yaml.YAMLError, AttributeError, KeyError):
        # 如果YAML解析失败，尝试用正则表达式提取
        try:
            # 匹配hosts字段，支持不同的格式
            host_pattern = r'^\s*-?\s*hosts:\s*(.+?)(?:\s*$|\s*#)'
            matches = re.findall(host_pattern, playbook_content, re.MULTILINE | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        except Exception:
            pass
        
        return 'unknown'


def should_use_limit_flag(playbook_content: str, target_host: str) -> bool:
    """
    判断是否应该使用-l参数来限制执行服务器
    
    Args:
        playbook_content: playbook的YAML内容
        target_host: 目标服务器
        
    Returns:
        bool: True表示应该使用-l参数，False表示不应该使用
    """
    hosts_value = extract_playbook_hosts(playbook_content)
    
    # 如果hosts是'all'或'localhost'，可以安全使用-l参数
    if hosts_value.lower() in ['all', 'localhost']:
        return True
    
    # 如果hosts是具体的服务器名或IP，检查是否与目标服务器匹配
    if hosts_value == target_host:
        return False  # 已经匹配，不需要-l参数
    
    # 如果hosts是服务器组名或其他值，为了安全起见不使用-l参数
    # 这种情况下依赖inventory和playbook中的hosts配置
    return False


async def select_playbooks(query: str, playbook_tools) -> list[dict]:
    """
    llm 推理决策执行剧本，并自动填写参数 - 非流式版本（保持兼容性）
    """
    logger.info(f"Input:\n{query}")

    playbook_list_str = '\n'.join([
        f"- {tool['name']}: {tool['description']} (文件名: {tool['file']})\n{tool['content']}" for tool in playbook_tools
    ])

    prompt = STRATEGY_PROMPT_TEMPLATE_v2.format(query=query, playbook_list=playbook_list_str)

    enable_thinking = config.get("strategy_think")
    if enable_thinking:
        response = query_llm(prompt, enable_thinking=enable_thinking)
        response = strip_think(response).strip()
    else:
        response = query_llm(prompt)

    selected_playbooks = json.loads(response)
    # 保留大模型推理的参数值，并生成命令
    for host, pb in selected_playbooks.items():
        if pb == "无合适剧本":
            continue
        pb_cmds = []
        for playbook in pb.get("playbooks", []):
            # 确保有 parameters 字段
            if "parameters" not in playbook:
                playbook["parameters"] = []
            
            # 构造命令，使用大模型推理的参数值
            param_str = ""
            if playbook["parameters"]:
                param_pairs = []
                for param in playbook["parameters"]:
                    if param.get("value"):  # 只使用有值的参数
                        param_pairs.append(f"{param['name']}={param['value']}")
                param_str = " ".join(param_pairs)
            
            # 查找对应的playbook工具以获取内容
            playbook_content = ""
            for tool in playbook_tools:
                if tool['file'] == playbook['file']:
                    playbook_content = tool.get('content', '')
                    break
            
            # 生成完整命令，根据playbook的hosts字段决定是否使用-l参数
            cmd = f"ansible-playbook -i /root/ansible/inventory"
            
            # 判断是否应该使用-l参数
            if should_use_limit_flag(playbook_content, host):
                cmd += f" -l {host}"
            
            cmd += f" /opt/thsre/{playbook['file']}"
            
            if param_str:
                cmd += f" -e \"{param_str}\""
            pb_cmds.append(cmd)
        
        # 保存生成的命令
        pb["commands"] = pb_cmds
    
    return selected_playbooks


async def select_playbooks_stream(query: str, playbook_tools) -> AsyncIterator[str]:
    """
    LLM 推理决策执行剧本，并自动填写参数 - 流式版本
    """
    logger.info(f"Input:\n{query}")

    playbook_list_str = '\n'.join([
        f"- {tool['name']}: {tool['description']} (文件名: {tool['file']})\n{tool['content']}" for tool in playbook_tools
    ])

    prompt = STRATEGY_PROMPT_TEMPLATE_v2.format(query=query, playbook_list=playbook_list_str)

    enable_thinking = config.get("strategy_think")
    show_think_block = config.get("strategy_show_think_chunk")

    # 只输出提示信息，然后收集LLM响应但不显示
    yield "\n**剧本选择中...**\n\n"

    # 收集LLM响应但不输出
    if show_think_block:
        # 如果配置要显示思考过程，则显示
        async for token in query_llm(
                prompt,
                stream = True,
                enable_thinking = enable_thinking,
                temperature = 0.1,
                top_p = 0.5,
                presence_penalty = 0,
                frequency_penalty = 0.3
            ):
            yield token
    else:
        # 不显示LLM的原始输出，只收集
        response_buffer = ""
        in_think = False
        tag_buf = ""
        end_buf = ""
        async for token in query_llm(
                prompt,
                stream = True,
                enable_thinking = enable_thinking,
                temperature = 0.1,
                top_p = 0.5,
                presence_penalty = 0,
                frequency_penalty = 0.3
            ):
            for c in token:
                if not in_think:
                    tag_buf += c
                    while tag_buf and not "<think>".startswith(tag_buf):
                        response_buffer += tag_buf[0]
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
            response_buffer += tag_buf
        
        # 返回收集到的响应（不显示给用户）
        yield response_buffer


async def execute_playbooks_stream(selected_playbooks: dict, executor_host: str = "mn21") -> AsyncIterator[str]:
    """
    流式执行剧本并输出结果
    """
    # 收集所有需要执行的命令
    pb_cmds_by_host = {}
    host_count = 0
    
    for host, pb in selected_playbooks.items():
        if pb == "无合适剧本":
            continue
            
        if "commands" in pb and pb["commands"]:
            pb_cmds_by_host[host] = {
                "commands": pb["commands"],
                "playbooks": pb.get("playbooks", [])
            }
            host_count += 1
    
    if not pb_cmds_by_host:
        return
    
    yield f"**开始执行剧本** (执行服务器: {executor_host})\n\n"
    # 执行命令并流式输出结果
    for host, data in pb_cmds_by_host.items():
        commands = data["commands"]
        playbooks = data["playbooks"]
        
        # yield f"### 🖥️ **目标服务器**: {host}\n\n"
        
        for i, (cmd, playbook) in enumerate(zip(commands, playbooks), 1):
            playbook_name = playbook.get('name', '未知剧本')
            playbook_file = playbook.get('file', 'unknown.yml')
            
            yield f"**执行剧本 {i}**: {playbook_name}, {playbook_file}\n"
            # yield f"**文件**: `{playbook_file}`\n\n"
            
            try:
                async for chunk in run_command_on_host(cmd, executor_host=executor_host):
                    yield chunk

            except Exception as e:
                error_msg = f"[Error] {type(e).__name__}: {str(e)}"
                yield f"**执行异常**: {error_msg}\n\n"
        
        yield "---\n\n"


async def handle_strategy_stream(exception_info: str, playbook_tools) -> AsyncIterator[str]:
    """
    处理异常信息并流式输出策略执行过程
    """
    try:
        data = json.loads(exception_info)
        query = data.get("query", "").strip()
    except json.JSONDecodeError:
        query = exception_info.strip()

    if not query:
        yield "**错误**: 无效的请求内容\n\n"
        return
    
    # 选择剧本并自动填写参数 - 流式输出
    response_buffer = ""
    
    async for chunk in select_playbooks_stream(query, playbook_tools):
        # 如果是提示信息，则显示；如果是LLM响应，则只收集
        if "**剧本选择中" in chunk:
            yield chunk
        else:
            response_buffer += chunk

    # 解析完整的JSON并格式化展示
    try:
        # 查找JSON内容
        json_start = response_buffer.find('{')
        json_end = response_buffer.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_buffer[json_start:json_end]
            selected_playbooks = json.loads(json_str)
            
            # 格式化展示剧本选择结果
            for host, pb in selected_playbooks.items():
                yield f"### Host: {host}\n\n"
                
                if pb == "无合适剧本":
                    yield "无合适剧本可执行\n\n"
                    yield "---\n\n"
                    continue
                
                playbooks = pb.get("playbooks", [])
                if not playbooks:
                    yield "无合适剧本可执行\n\n"
                    yield "---\n\n"
                    continue
                    
                for i, playbook in enumerate(playbooks, 1):
                    yield f"**剧本 {i}**\n"
                    yield f"- **名称**: {playbook.get('name', 'N/A')}\n"
                    yield f"- **描述**: {playbook.get('description', 'N/A')}\n"
                    yield f"- **文件**: {playbook.get('file', 'N/A')}\n"
                    
                    parameters = playbook.get('parameters', [])
                    if parameters:
                        yield "- **参数**:\n"
                        for param in parameters:
                            param_name = param.get('name', 'N/A')
                            param_type = param.get('type', 'N/A')
                            param_value = param.get('value', 'N/A')
                            yield f"  - `{param_name}` ({param_type}): {param_value}\n"
                    else:
                        yield "- **参数**: 无需参数\n"
                    
                    yield "\n"  # 剧本之间的空行
                
                yield "---\n\n"
            
            # 为每个服务器生成命令
            for host, pb in selected_playbooks.items():
                if pb == "无合适剧本":
                    continue
                pb_cmds = []
                for playbook in pb.get("playbooks", []):
                    if "parameters" not in playbook:
                        playbook["parameters"] = []
                    
                    param_str = ""
                    if playbook["parameters"]:
                        param_pairs = []
                        for param in playbook["parameters"]:
                            if param.get("value"):
                                param_pairs.append(f"{param['name']}={param['value']}")
                        param_str = " ".join(param_pairs)
                    
                    # 查找对应的playbook工具以获取内容
                    playbook_content = ""
                    for tool in playbook_tools:
                        if tool['file'] == playbook['file']:
                            playbook_content = tool.get('content', '')
                            break
                    
                    # 生成完整命令，根据playbook的hosts字段决定是否使用-l参数
                    cmd = f"ansible-playbook -i /root/ansible/inventory"
                    
                    # 判断是否应该使用-l参数
                    if should_use_limit_flag(playbook_content, host):
                        cmd += f" -l {host}"
                    
                    cmd += f" /opt/thsre/{playbook['file']}"
                    
                    if param_str:
                        cmd += f" -e \"{param_str}\""
                    pb_cmds.append(cmd)
                
                pb["commands"] = pb_cmds
            
            logger.info(f"Selected playbooks with params:\n{selected_playbooks}")
            
            # 流式执行剧本
            async for chunk in execute_playbooks_stream(selected_playbooks):
                yield chunk
                
        else:
            yield "**错误**: 无法解析剧本选择结果\n\n"
            
    except json.JSONDecodeError as e:
        yield f"**JSON解析错误**: {str(e)}\n\n"
        yield f"**原始响应**: {response_buffer}\n\n"
    except Exception as e:
        yield f"**执行错误**: {str(e)}\n\n"


async def _send(client: A2AClient, query):
    payload = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": query}],
            "messageId": uuid.uuid4().hex,
        },
    }

    req = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams(**payload))
    resp = await client.send_message(req)
    result = resp.model_dump(mode="json", exclude_none=True)

    return result["result"]["parts"][0]["text"]


async def run_command_on_host(command: str, executor_host: str = "mn21"):
    base_url = f"http://{executor_host}:5004"

    async with httpx.AsyncClient(verify=False, timeout=3000, proxy=None) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url,)
        final_agent_card_to_use = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)

        result = await _send(client, command)
        yield parse_playbook_output(result)


def parse_playbook_output(output: str) -> str:
    """
    解析 ansible-playbook 的输出，提取关键信息
    成功时：返回 TASK [debug] 中 ok 后面 "msg" 里的内容
    失败时：返回 fatal 中的错误信息
    """
    lines = output.strip().split('\n')
    
    # 检查是否有 fatal 错误
    for line in lines:
        if 'fatal:' in line and '=>' in line:
            # 提取 fatal 后面的 JSON 内容
            try:
                json_start = line.find('=>') + 2
                json_part = line[json_start:].strip()
                if json_part.startswith('{') and json_part.endswith('}'):
                    error_data = json.loads(json_part)
                    return error_data.get('msg', json_part)
                else:
                    # 如果不是标准 JSON 格式，返回整个 fatal 行
                    return line.strip()
            except json.JSONDecodeError:
                return line.strip()
    
    # 查找成功情况下的 debug 信息
    in_debug_task = False
    for i, line in enumerate(lines):
        # 检测 TASK [debug] 开始
        if 'TASK [debug]' in line:
            in_debug_task = True
            continue
        
        # 在 debug task 中查找 ok 行
        if in_debug_task and 'ok:' in line and '=>' in line:
            try:
                # 提取 ok 后面的 JSON 内容
                json_start = line.find('=>') + 2
                json_part = line[json_start:].strip()
                
                # 处理多行的情况
                full_json = json_part
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('}'):
                    full_json += '\n' + lines[j]
                    j += 1
                if j < len(lines):
                    full_json += '\n' + lines[j]
                
                # 解析 JSON 并提取 msg
                if full_json.strip().startswith('{') and '}' in full_json:
                    try:
                        data = json.loads(full_json.strip())
                        msg = data.get('msg', [])
                        if isinstance(msg, list):
                            return '\n'.join(msg)
                        else:
                            return str(msg)
                    except json.JSONDecodeError:
                        # 如果 JSON 解析失败，尝试用正则提取 msg 内容
                        msg_match = re.search(r'"msg":\s*\[(.*?)\]', full_json, re.DOTALL)
                        if msg_match:
                            msg_content = msg_match.group(1)
                            # 清理引号和逗号，提取实际内容
                            lines_content = re.findall(r'"([^"]*)"', msg_content)
                            return '\n'.join(lines_content)
            except Exception:
                pass
        
        # 如果遇到下一个 TASK 或 PLAY，停止查找
        if in_debug_task and ('TASK [' in line or 'PLAY [' in line) and 'TASK [debug]' not in line:
            break
    
    # 如果都没找到，返回原始输出
    return f"```bash\n{output}\n```\n\n"


async def execute_on_multiple_hosts(pb_cmds_by_host: dict, executor_host: str = "mn21") -> dict:
    # 所有命令都在executor_host依次执行，分段收集结果
    all_cmds = []
    for commands in pb_cmds_by_host.values():
        all_cmds.extend(commands)
    results = []
    for cmd in all_cmds:
        try:
            output = await run_command_on_host(cmd, executor_host=executor_host)
            # 解析 playbook 输出，只保留关键信息
            parsed_output = parse_playbook_output(output)
            results.append({"cmd": cmd, "output": parsed_output})
        except Exception as e:
            results.append({"cmd": cmd, "output": f"[Error] {type(e).__name__}: {str(e)}"})
    return {executor_host: results}


async def handle_payload(exception_info: str, playbook_tools) -> dict:
    """处理异常信息并执行剧本"""
    try:
        data = json.loads(exception_info)
        query = data.get("query", "").strip()
    except json.JSONDecodeError:
        query = exception_info.strip()

    if not query:
        return {"status": "error", "msg": "无效的请求内容"}

    # 选择剧本并自动填写参数
    selected_playbooks = await select_playbooks(query, playbook_tools)
    logger.info(f"Selected playbooks with params:\n{selected_playbooks}")

    # 收集所有需要执行的命令
    pb_cmds_by_host = {}
    for host, pb in selected_playbooks.items():
        if pb == "无合适剧本":
            continue
        if "commands" in pb:
            pb_cmds_by_host[host] = pb["commands"]

    # 执行命令并返回结果（所有命令都在executor_host依次执行，分段输出）
    executor_host = "mn21"  # 后续可通过配置或参数传入
    if pb_cmds_by_host:
        results = await execute_on_multiple_hosts(pb_cmds_by_host, executor_host=executor_host)
        output_blocks = []
        for item in results[executor_host]:
            output_blocks.append(f"$ {item['cmd']}\n{item['output']}\n")
        combined_output = f"【{executor_host}】\n" + "\n".join(output_blocks)
        logger.info(f"执行结果：\n{combined_output}")
        return {"status": "success", "result": combined_output}

    # if pb_cmds_by_host:
    #     return {"status": "success", "result": pb_cmds_by_host}
    
    # 如果没有要执行的命令，返回剧本选择结果
    return {"status": "success", "result": selected_playbooks}


# --- 8< --- 业务层 --- 8< ---
class StrategyExecutor(AgentExecutor):
    def __init__(self):
        self.m_client = PlaybookMilvusClient()

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        user_input = context.get_user_input()
        playbooks = self.m_client.search_playbooks_by_desc(user_input, top_k=20)
        
        # 使用流式输出
        async for chunk in handle_strategy_stream(user_input, playbooks):
            await event_queue.enqueue_event(new_agent_text_message(chunk))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception('cancel not supported')


# --- 8< --- Agent Card --- 8< ---
skill = AgentSkill(
    id = "strategy_plan",
    name = "Strategy plan",
    description = "根据系统状态数据和异常分析情况制定自愈策略",
    tags = []
)

# 获取真实服务器名
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
                m_client = PlaybookMilvusClient()
                playbooks = m_client.search_playbooks_by_desc(query, top_k=20)
                
                async for token in handle_strategy_stream(query, playbooks):
                    if token:
                        chunk = {
                            "id": str(uuid.uuid4()),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "strategy-agent",
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
                    "model": "strategy-agent",
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
            "http://localhost:5003/a2a/v1/agent/messages",
            content=await request.body(),
            headers=dict(request.headers)
        )
        return response.json()


def main() -> None:
    # 启动合并的服务器（同时支持A2A和HTTP流式）
    uvicorn.run(combined_app, host="0.0.0.0", port=5003)


if __name__ == '__main__':
    main()

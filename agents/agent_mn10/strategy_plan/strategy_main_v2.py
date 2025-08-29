import os, sys, json
sys.path.append('./')
from pathlib import Path
from agentsAPI import query_llm, strip_think
import socket, uvicorn, logging
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.tasks import InMemoryTaskStore
from milvus_client import PlaybookMilvusClient
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    AgentCapabilities,
    AgentSkill
)
import yaml, httpx, uuid, re


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


STRATEGY_SELECT_PLAYBOOK_PROMPT = """你是自动化运维专家。你将收到一份诊断报告，包括一台或多台服务器的**异常描述、已执行命令及其输出、异常分析**，以及针对这些异常可用的剧本库（包含剧本基本信息和所需参数）。

你的任务：
1. **意图识别**：分析诊断信息，识别所有需要处理的主机、任务类型、异常现象和修复目标。对每台服务器，**独立分析**其异常根因，确保聚焦于主机问题，能区分多主机/多任务场景，按主机/任务分别处理。
2. **剧本筛选**：针对每个主机/任务，独立判断是否有合适的剧本。**精确匹配**与异常直接相关、能解决当前问题的1~2个剧本，能修好为止。合理安排剧本执行顺序，**避免无关或重复修复步骤**。

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
                    {{"name": "参数名", "type": "参数类型", "value": "推理出的参数值"}},
                    ...
                ]
            }}
        ],
        ...
    }},
    "host2": "无合适剧本"
}}

**注意：**
- 必须先做意图识别，分清主机/任务，再筛选剧本。
- 剧本匹配必须基于异常本身、可用剧本描述，**不允许主观臆断或推测缺失信息**。
- 无需考虑剧本中的 "hosts" 字段，**只需关注剧本的功能和参数**。
- **剧本少优于多，相关优于覆盖**。
- **只输出标准 JSON，绝不添加注释、说明、理由或多余文字**。

当前系统的诊断信息（含已执行命令、异常输出、部分分析）：
{query}

与异常描述相关的可选 ansible 剧本列表：
{playbook_list}
"""


STRATEGY_PARAM_INFER_PROMPT = """你是自动化运维专家。你将收到一个已选定的剧本（含参数定义）和诊断信息。

你的任务：
- 只针对该剧本，结合剧本参数定义和诊断信息，自动推理并填写所有必需参数。
- 你需要仔细分析剧本参数的含义和用途，从主机状态、异常信息、已执行命令上下文中提取相关值，按照参数类型给出合理的值。
- 若无法推理某参数，值留空即可。

输出格式严格如下，内容为**JSON**，不包含任何注释、说明、理由或多余文字：
[
    {{
        "name": "参数名",
        "type": "参数类型",
        "value": "推理出的参数值"
    }},
    ...
]

剧本参数：
{param_def}

诊断信息：
{query}
"""


GEN_PLAYBOOK_PROMPT_TEMPLATE = """你是一名资深自动化运维工程师。

请根据以下异常诊断信息，现场编写一个最合适的 Ansible 剧本（YAML 格式）或运维 Shell 脚本。

要求：
- 能通过剧本/脚本直接修复该异常，剧本需包含详细注释、参数定义，且必须为可直接执行的标准；
- 仅输出剧本具体内容，不要有任何说明或多余文字。

异常诊断信息：
{query}
"""


GEN_PLAYBOOK_FILENAME_PROMPT = """你是一名资深自动化运维工程师。

请根据下面的异常描述和剧本内容，为该剧本生成一个简洁、有实际意义的英文文件名（以 .yml 结尾，全部小写，单词用下划线分隔，能体现主要修复对象或动作，避免使用通用词）。

只输出文件名本身，不要输出其它内容。

异常描述：
{query}

剧本内容：
{playbook_content}
"""


async def select_playbooks(query: str, playbook_tools) -> list[dict]:
    """
    llm 推理决策执行剧本，并自动填写参数
    """
    logger.info(f"Input：\n{query}")

    playbook_list_str = '\n'.join([
        f"- {tool['name']}: {tool['description']} (文件名: {tool['file']})\n{tool['content']}" for tool in playbook_tools
    ])
    prompt = STRATEGY_SELECT_PLAYBOOK_PROMPT.format(query=query, playbook_list=playbook_list_str)
    enable_thinking = config.get("strategy_think")
    if enable_thinking:
        response = query_llm(prompt, enable_thinking=enable_thinking)
        response = strip_think(response).strip()
    else:
        response = query_llm(prompt)
    selected_playbooks = json.loads(response)
    return selected_playbooks


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


# 通用自动化执行器接口
class AutomationExecutor:
    def build_command(self, host: str, playbook: dict) -> str:
        pb_type = playbook.get("type", "ansible")
        param_str = ""
        if playbook.get("parameters"):
            param_pairs = []
            for param in playbook["parameters"]:
                if param.get("value"):
                    param_pairs.append(f"{param['name']}={param['value']}")
            param_str = " ".join(param_pairs)
        if pb_type == "ansible":
            cmd = f"ansible-playbook -i /root/ansible/inventory -l {host} /opt/thsre/{playbook['file']}"
            if param_str:
                cmd += f" -e \"{param_str}\""
            return cmd
        elif pb_type == "shell":
            # shell 脚本执行方式
            cmd = f"bash /opt/thsre/{playbook['file']} {param_str}".strip()
            return cmd
        elif pb_type == "api":
            # API 调用方式（这里只是示例，具体实现需补充）
            return f"curl -X POST http://{host}/api/{playbook['file']} -d '{param_str}'"
        else:
            # 其他类型可扩展
            return f"echo 'Unsupported playbook type: {pb_type}'"


    async def run_command(self, command: str, executor_host: str = "mn21") -> str:
        base_url = f"http://{executor_host}:5004"

        async with httpx.AsyncClient(verify=False, timeout=3000, proxy=None) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url,)
            final_agent_card_to_use = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)

            result = await _send(client, command)
            return result


    async def execute_on_multiple_hosts(self, pb_cmds_by_host: dict, executor_host: str = "mn21") -> dict:
        all_cmds = []
        for commands in pb_cmds_by_host.values():
            all_cmds.extend(commands)
        results = []
        for cmd in all_cmds:
            try:
                output = await self.run_command(cmd, executor_host=executor_host)
                results.append({"cmd": cmd, "output": output.strip()})
            except Exception as e:
                results.append({"cmd": cmd, "output": f"[Error] {type(e).__name__}: {str(e)}"})
        return {executor_host: results}


async def handle_payload(exception_info: str, m_client: PlaybookMilvusClient) -> dict:
    """处理异常信息并执行剧本（兼容多种执行方式）"""
    executor = AutomationExecutor()
    try:
        data = json.loads(exception_info)
        query = data.get("query", "").strip()
        confirmed_params = data.get("confirmed_params")  # 获取已确认的参数
    except json.JSONDecodeError:
        query = exception_info.strip()
        confirmed_params = None

    if not query:
        return {"status": "error", "msg": "无效的请求内容"}

    if confirmed_params:
        # 用户已确认参数，拼接命令并执行（支持多种类型）
        pb_cmds_by_host = {}
        for host, pb_list in confirmed_params.items():
            cmds = []
            for playbook in pb_list:
                cmd = executor.build_command(host, playbook)
                cmds.append(cmd)
            if cmds:
                pb_cmds_by_host[host] = cmds
        executor_host = "mn21"
        if pb_cmds_by_host:
            results = await executor.execute_on_multiple_hosts(pb_cmds_by_host, executor_host=executor_host)
            output_blocks = []
            for item in results[executor_host]:
                output_blocks.append(f"$ {item['cmd']}\n{item['output']}\n")
            combined_output = f"【{executor_host}】\n" + "\n".join(output_blocks)
            return {
                "status": "success",
                "result": combined_output
            }
        return {
            "status": "success",
            "result": pb_cmds_by_host
        }

    # 循环检索-生成-检索，直到有合适剧本
    enable_thinking = config.get("strategy_think")
    while True:
        playbooks = m_client.search_playbooks_by_desc(query, top_k=20)
        selected_playbooks = await select_playbooks(query, playbooks)
        logger.info(f"Selected playbooks: {selected_playbooks}")
        # 判断是否所有主机都无合适剧本
        if isinstance(selected_playbooks, dict) and all((v == "无合适剧本" or v is None or v == []) for v in selected_playbooks.values()):
            # 现场生成新剧本
            prompt = GEN_PLAYBOOK_PROMPT_TEMPLATE.format(query=query)
            if enable_thinking:
                new_playbook_content = query_llm(prompt, enable_thinking=enable_thinking)
                new_playbook_content = strip_think(new_playbook_content).strip()
            else:
                new_playbook_content = query_llm(prompt)

            # 用LLM生成合适的英文文件名（中文提示）
            filename_prompt = GEN_PLAYBOOK_FILENAME_PROMPT.format(query=query, playbook_content=new_playbook_content)
            file_name = query_llm(filename_prompt).strip()

            file_name = file_name.replace(' ', '_').replace('-', '_').lower()
            if not file_name.endswith('.yml'):
                file_name += '.yml'
            file_name = re.sub(r'[^a-z0-9_\.]', '', file_name)
            playbook_dir = "/opt/thsre"
            file_path = os.path.join(playbook_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_playbook_content.strip())

            try:
                m_client.add_playbook(file_path)
            except Exception as e:
                logger.error(f"自动入库新剧本失败: {e}")
                return {"status": "error", "msg": f"自动生成剧本并入库失败: {e}"}
            continue
        else:
            break


    playbooks_with_params = {}
    for host, pb_list in selected_playbooks.items():
        if pb_list == "无合适剧本" or pb_list is None or pb_list == []:
            continue
        playbooks_with_params[host] = []
        for pb in pb_list:
            # 获取剧本参数定义
            pb_info = m_client.search_playbooks_by_file(pb["file"])
            param_def = pb_info.get("parameters", [])
            if isinstance(param_def, str):
                try:
                    param_def = yaml.safe_load(param_def)
                except Exception:
                    param_def = []
            if not isinstance(param_def, list):
                param_def = []
            # 用LLM推理参数
            param_prompt = STRATEGY_PARAM_INFER_PROMPT.format(param_def=json.dumps(param_def, ensure_ascii=False), query=query)
            if enable_thinking:
                param_json = query_llm(param_prompt, enable_thinking=enable_thinking)
                param_json = strip_think(param_json).strip()
            else:
                param_json = query_llm(param_prompt).strip()
            try:
                params = json.loads(param_json)
            except Exception:
                params = []
            pb_with_param = dict(pb)
            pb_with_param["parameters"] = params
            playbooks_with_params[host].append(pb_with_param)

    # 执行剧本
    pb_cmds_by_host = {}
    for host, pb_list in playbooks_with_params.items():
        cmds = []
        for playbook in pb_list:
            cmd = executor.build_command(host, playbook)
            cmds.append(cmd)
        if cmds:
            pb_cmds_by_host[host] = cmds
    executor_host = "mn21"
    if pb_cmds_by_host:
        results = await executor.execute_on_multiple_hosts(pb_cmds_by_host, executor_host=executor_host)
        output_blocks = []
        for item in results[executor_host]:
            output_blocks.append(f"$ {item['cmd']}\n{item['output']}\n")
        combined_output = f"【{executor_host}】\n" + "\n".join(output_blocks)
        return {
            "status": "success",
            "result": combined_output
        }
    return {
        "status": "success",
        "result": pb_cmds_by_host
    }


# --- 8< --- 业务层 --- 8< ---
class StrategyExecutor(AgentExecutor):
    def __init__(self):
        self.m_client = PlaybookMilvusClient()


    async def execute(self, context: RequestContext, event_queue: EventQueue):
        try:
            user_input = context.get_user_input()

            try:
                input_data = json.loads(user_input)
            except json.JSONDecodeError:
                input_data = {"query": user_input}

            result = await handle_payload(json.dumps(input_data), self.m_client)
            await event_queue.enqueue_event(new_agent_text_message(json.dumps(result, ensure_ascii=False)))

        except Exception as e:
            error_result = {"status": "error", "msg": f"执行错误: {str(e)}"}
            await event_queue.enqueue_event(new_agent_text_message(json.dumps(error_result, ensure_ascii=False)))


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

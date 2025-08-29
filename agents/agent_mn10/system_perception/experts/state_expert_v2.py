import os, sys, re
sys.path.append('./')
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


import yaml
import logging
import uuid, json
import httpx, asyncio
from pathlib import Path
from collections import defaultdict
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

# 尝试导入工具管理器
try:
    from agent_mn10.system_perception.experts.tools import tool_manager
except ImportError:
    # 如果导入失败，创建一个空的工具管理器
    class MockToolManager:
        def find_matching_tools(self, task):
            return []
        def call_tool(self, tool_name):
            return []
    tool_manager = MockToolManager()

from agentsAPI import query_llm, get_rag_rpc_only_content, strip_think, get_rag_rpc


agents_dir = Path(__file__).resolve().parents[3]
config_path = agents_dir / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


logging.basicConfig(
    filename = os.path.join(log_dir, "system_perception_agent.log"),
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.propagate = False


PROMPT_TEMPLATE = """你是一个资深系统运维专家，负责维护多个核心服务器集群。你的任务是根据【用户查询】，输出“服务器名称 -> 命令列表”的 JSON 字典，用于并发执行状态查询或诊断任务。

## 工作流程（严格按顺序执行）
1. **意图解析**
    - 逐句阅读【用户查询】，识别所有提到的服务器（支持别名、标签）。
    - 提取每台服务器需要执行的高层操作意图（如“查看磁盘使用”，“检测 Nginx 进程”）。

2. **直接命令映射**
    - 尝试将每个意图直接映射为最恰当、最简洁且安全的 shell 命令。
    - 优先使用行业惯用、可移植的命令行工具（如 `df -h`, `ps -ef | grep nginx`）。

3. **上下文替换优化**
    - 查看【上下文】与【已生成命令】，判断是否已有能满足相同功能且更贴合当前服务器环境、或语义更清晰的替代命令。
    - 若有更优替换：
        - 用新命令替换旧命令；
        - 确保功能等价或更好。
    - 若暂无更优替换，则保留原命令。

4. **结果去重与校验**
    - 删除同一服务器下功能重复的命令。
    - 确保命令无副作用（禁止 `rm`, `kill`, `reboot`, `shutdown`, `:(){{ :|:& }};:` 等危险操作）。
    - 禁止使用交互式或持续输出类命令，如 `top`, `htop`, `watch`, `tail -f`, `journalctl -f`。
    - 确保每条命令可直接在 Bash 终端执行，不含嵌套、无多余空格或换行。

5. **输出**
    - 仅输出 JSON 字典，格式示例：
        {{
            "hostname1": ["cmd1", "cmd2"],
            "hostname2": ["cmd1"]
        }}
    - 不要添加任何说明、注释或 Markdown。

【上下文】
{context}

【已生成命令】
{existing_commands}

【用户查询】
{query}
"""


class StateExpert():
    """自然语言 -> shell 命令 -> 终端执行 -> 获取输出"""
    def __init__(self, config):
        self.safe_commands = config.get("safe_commands", [])
        self.sensitive_keywords = config.get("sensitive_keywords", [])
        self.tool_manager = tool_manager


    def parse_query_with_tools(self, query: str) -> dict[str, list[str]]:
        """
        使用语义相似度匹配来找到合适的工具命令
        """
        host_cmds = defaultdict(list)
        
        # 1. 匹配服务器名称
        host_blocks = re.findall(r"获取\s+([\w\-]+)\s*服务器的([^。；，]*)", query)
        
        for hostname, task_block in host_blocks:
            # 2. 提取任务描述
            task_phrases = re.findall(r"【([^】]+)】", task_block)
            
            for task in task_phrases:
                # 3. 使用语义相似度找到最匹配的工具
                matching_tools = self.tool_manager.find_matching_tools(task)
                
                # 4. 调用匹配度最高的工具
                for tool_name, similarity in matching_tools:
                    cmds = self.tool_manager.call_tool(tool_name)
                    logger.info(f"匹配工具: {tool_name} (相似度: {similarity:.3f}) -> {cmds}")
                    host_cmds[hostname].extend(cmds)
        
        return dict(host_cmds)


    def gen_shell(self, query: str, source: list, need_rag: bool = False) -> dict[str, list[str]]:
        """
        根据 query 执行关键词匹配 + LLM，合成命令字典
        返回结构: {hostname: [cmd1, cmd2]}
        """
        # 1. 提取工具命令
        tool_cmd_dict = self.parse_query_with_tools(query)

        # 2. 获取上下文
        context = get_rag_rpc_only_content(query) if need_rag else ""
        # context = get_rag_rpc(query, source) if need_rag else ""

        # 3. 构造 LLM Prompt
        prompt = PROMPT_TEMPLATE.format(
            query = query,
            context = context,
            existing_commands = json.dumps(tool_cmd_dict, ensure_ascii=False, indent=2) if tool_cmd_dict else "{}"
        )

        # 4. 调用 LLM 生成命令
        enable_thinking = config.get("gen_shell_think")
        if enable_thinking:
            response = query_llm(prompt, enable_thinking=enable_thinking)
            response = strip_think(response).strip()
        else:
            response = query_llm(prompt)

        cleaned = self.clean_script(response)

        try:
            llm_cmd_dict = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"❌ LLM 输出无法解析为 JSON，内容如下：\n{cleaned}")
            raise ValueError("LLM 输出格式不正确，无法转换为命令字典") from e

        # 5. 合并 LLM 和工具命令（去重）
        merged_cmds = defaultdict(list)
        for hostname, cmds in tool_cmd_dict.items():
            merged_cmds[hostname].extend(cmds)

        for hostname, llm_cmds in llm_cmd_dict.items():
            if not isinstance(llm_cmds, list):
                logger.warning(f"跳过异常 host: {hostname}，值不是 list")
                continue
            merged = set(merged_cmds.get(hostname, []))
            for cmd in llm_cmds:
                if cmd not in merged:
                    merged_cmds[hostname].append(cmd)

        return dict(merged_cmds)


    def clean_script(self, script: str) -> str:
        """
        对 LLM 生成的命令进行格式化，满足输入终端可执行的命令格式
        """
        pattern = r'```[\s\S]*?\n(.*?)\n```'
        matches = re.findall(pattern, script, re.DOTALL)
        cleaned_content = "\n".join(match.strip() for match in matches)
        return cleaned_content if cleaned_content else script.strip()


    def sandbox_check_batch(self, shell_cmds_by_host: dict[str, list[str]]):
        def _check_one_host_cmds(cmds: list[str], host: str):
            for cmd in cmds:
                normalized_cmd = cmd.lower().strip()

                # 白名单放行（严格匹配）
                if any(normalized_cmd.startswith(safe + " ") or normalized_cmd == safe for safe in self.safe_commands):
                    continue

                # 检查敏感关键词
                for keyword in self.sensitive_keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', normalized_cmd):
                        raise ValueError(f"沙盒检测失败：在【{host}】的命令【{cmd}】中发现关键词【{keyword}】")

                # 检查破坏性参数
                if re.search(r'\s-\w*f\b', normalized_cmd) or "--force" in normalized_cmd:
                    raise ValueError(f"沙盒检测失败：在【{host}】的命令【{cmd}】中检测到强制执行选项")

                # 审计日志
                logger.warning(f"沙盒警告：命令【{cmd}】（来自【{host}】）不在白名单中，需额外审查")

        for host, cmds in shell_cmds_by_host.items():
            _check_one_host_cmds(cmds, host)


    async def _send(self, client: A2AClient, query):
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


    async def run_command_on_host(self, hostname, command: str) -> str:
        base_url = f"http://{hostname}:5004"

        async with httpx.AsyncClient(verify=False, timeout=300, proxy=None) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url,)
            final_agent_card_to_use = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)

            result = await self._send(client, command)
            return result


    async def execute_on_multiple_hosts(self, shell_cmds_by_host: dict) -> dict:
        tasks = []
        hostnames = []

        for hostname, commands in shell_cmds_by_host.items():
            joined_cmd = "; ".join(commands)
            tasks.append(self.run_command_on_host(hostname, joined_cmd))
            hostnames.append(hostname)

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for hostname, result in zip(hostnames, raw_results):
            if isinstance(result, Exception):
                # 保留错误信息作为字符串
                results[hostname] = f"[Error] {type(result).__name__}: {str(result)}"
            else:
                results[hostname] = result.strip()

        return results


    async def get_system_state(self, query: str) -> dict:
        logger.info(f"输入查询：{query}")

        try:
            # 1. 使用 LLM 生成多台服务器命令
            source = ['slurm', 'lustre', 'system']
            need_rag = config.get("gen_shell_need_rag")
            shell_cmds_by_host = self.gen_shell(query, source, need_rag)
            logger.info(f"生成的分发命令：{shell_cmds_by_host}")

            # 2. 黑白名单筛选 + 并发执行命令
            self.sandbox_check_batch(shell_cmds_by_host)

            # 3. 并发执行：访问各服务器的执行 Agent
            results = await self.execute_on_multiple_hosts(shell_cmds_by_host)

            # 4. 拼接成结构化字符串
            combined_output = "\n\n".join(
                f"### {host}\n```\n{output}\n```" for host, output in results.items()
            )
            logger.info(f"执行结果汇总：{combined_output}")

            # 保证 result 字段为 dict，便于 merge_result 合并
            return {"status": "success", "result": {"output": combined_output}}

        except Exception as e:
            logger.error(f"SystemState Error: {str(e)}")
            return {"status": "error", "result": {"message": str(e)}}


    async def get_system_state_stream(self, query: str):
        """
        流式版本的系统状态获取，逐步返回执行结果
        """
        # logger.info(f"流式输入查询：{query}")

        try:
            # 1. 使用 LLM 生成多台服务器命令
            source = ['slurm', 'lustre', 'system']
            need_rag = config.get("gen_shell_need_rag")
            shell_cmds_by_host = self.gen_shell(query, source, need_rag)
            logger.info(f"生成的分发命令：{shell_cmds_by_host}")

            # 输出命令生成阶段
            yield {"status": "progress", "message": "### 命令生成完成", "commands": shell_cmds_by_host}

            # 2. 黑白名单筛选
            self.sandbox_check_batch(shell_cmds_by_host)
            yield {"status": "progress", "message": "命令安全检查通过"}

            # 3. 逐台服务器执行并输出结果
            final_results = {}
            for hostname, commands in shell_cmds_by_host.items():
                try:
                    yield {"status": "progress", "message": f"正在执行 {hostname} 上的命令..."}
                    
                    joined_cmd = "; ".join(commands)
                    result = await self.run_command_on_host(hostname, joined_cmd)
                    final_results[hostname] = result.strip()
                    
                    # 实时输出单台服务器的结果
                    yield {
                        "status": "progress", 
                        "message": f"{hostname} 执行完成",
                        "host_result": {hostname: final_results[hostname]}
                    }
                    
                except Exception as e:
                    error_msg = f"[Error] {type(e).__name__}: {str(e)}"
                    final_results[hostname] = error_msg
                    yield {
                        "status": "progress",
                        "message": f"{hostname} 执行出错: {str(e)}",
                        "host_result": {hostname: error_msg}
                    }

            # 4. 输出最终汇总结果
            combined_output = "\n\n".join(
                f"### {host}\n```\n{output}\n```" for host, output in final_results.items()
            )
            
            yield {
                "status": "success", 
                "result": {"output": combined_output},
                "message": "所有服务器执行完成"
            }

        except Exception as e:
            logger.error(f"SystemState Stream Error: {str(e)}")
            yield {
                "status": "error", 
                "result": {"message": str(e)},
                "error_message": str(e)
            }

import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:google.protobuf.runtime_version'
import re
import sys
import json
import time
import uuid
import httpx
import asyncio
import argparse
import markdown
import traceback
from bs4 import BeautifulSoup
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')


HOST = "a6000-G5500-V6"
PORT = 5001
# task = "查看节点 cn79873 上【作业 20250625 的运行情况】、【slurmd 服务的运行情况】和【重启 slurmd 服务】，如果存在异常，需要分析并给出报告"
# task = "Lustre 文件系统如何通过设置精细控制 Changelog 日志的采集范围？在哪些场景下需要这样做？"
# task = "服务器 a6000 的GPU指标出现异常，进行异常情况分析"
# task = "节点 cn79873 的磁盘出现故障，进行故障分析和根因定位"
# task = "a6000-G5500-V6 的 rca_api 服务出现异常，制定修复策略、执行修复并验证结果"


async def _send(client: A2AClient, query):
    if isinstance(query, dict):
        text_part = json.dumps(query, ensure_ascii=False)
    else:
        text_part = str(query)

    payload = {
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": text_part}],
            "messageId": uuid.uuid4().hex,
        },
    }
    # print(f"🔗 Connecting to Agent...")
    req = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams(**payload))

    # print(f"🤖 Agent executing...")
    resp = await client.send_message(req)

    result = resp.model_dump(mode="json", exclude_none=True)
    return result["result"]["parts"][0]["text"]


def pretty_print_output(text: str) -> None:
    """
    解析外层 JSON 字符串，输出主机名以及完整命令结果，
    仅做最基本的换行/分隔处理，不再区分日志类型。
    """
    try:
        # 1️⃣ 外层 JSON 反序列化，得到真正字符串（反转义后的 \n）
        try:
            content = json.loads(text)
        except json.JSONDecodeError:
            content = text

        # 2️⃣ 提取主机名（形如【hostname】）
        host = "未知主机"
        if content.startswith("【"):
            right = content.find("】")
            if right != -1:
                host = content[1:right]
                content = content[right + 1:].lstrip("\n")

        # 3️⃣ 打印
        print(f"\033[1;35m{'-' * 30} Command outputs {'-' * 30}\033[0m")
        print(f"\033[1;33m[{host}]\033[0m\n")

        # ★ 关键：此时 content 已经是正常换行字符串，直接输出即可
        # 如果觉得需要再在每个 `$` 开头前加分隔，可再 split/print
        blocks = re.split(r'(?=^\$ )', content, flags=re.MULTILINE)
        for blk in blocks:
            blk = blk.rstrip()
            if not blk:
                continue
            print(blk, "\n")

    except Exception as e:
        print(f"❌ pretty_print_output error: {e}")
        traceback.print_exc()


def extract_plain_text(response_text):
    """从响应中提取纯文本内容"""
    try:
        # 解析JSON响应
        data = json.loads(response_text)
        
        # 提取内容字符串
        if "choices" in data and isinstance(data["choices"], list):
            content = data["choices"][0]["message"]["content"]
        elif isinstance(data, dict) and len(data) == 1 and isinstance(list(data.values())[0], str):
            content = list(data.values())[0]
        else:
            return response_text  # 无法解析则返回原始文本
        
        # 尝试解析内层JSON（如果存在）
        try:
            inner_data = json.loads(content)
            if isinstance(inner_data, dict):
                # 取第一个值作为内容
                content = list(inner_data.values())[0]
        except json.JSONDecodeError:
            pass  # 不是JSON则继续使用原内容
        
        # 移除Markdown代码块标记
        content = re.sub(r'```(markdown)?\s*', '', content)
        
        # 移除Markdown格式的标题、列表等标记
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)  # 标题
        content = re.sub(r'^\s*[-*]\s*', '', content, flags=re.MULTILINE)  # 列表项
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # 加粗
        content = re.sub(r'_(.*?)_', r'\1', content)  # 斜体
        
        # 移除多余的空行
        content = re.sub(r'\n\s*\n', '\n\n', content).strip()
        
        return content
    except Exception as e:
        print(f"❌ 解析响应时出错: {e}")
        traceback.print_exc()
        return response_text


async def agent_quest(query: str):
    base_url = f"http://{HOST}:{PORT}"
    try:
        async with httpx.AsyncClient(verify=False, timeout=300, proxy=None) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            final_agent_card_to_use = await resolver.get_agent_card()
            client = A2AClient(
                httpx_client=httpx_client, agent_card=final_agent_card_to_use
            )
            result = await _send(client, query)
            plain_text = extract_plain_text(result)

            # 检查是否需要二次确认
            resp_json = json.loads(plain_text)
            if resp_json.get("status") == "need_confirm":
                to_confirm = resp_json.get("to_confirm", {})
                cmds_for_confirm = resp_json.get("cmds_for_confirm", {})
                print("\nNeed confirm commands:")
                confirm_dict = {}
                for host, cmds in to_confirm.items():
                    print(f"\n【{host}】:")
                    for cmd in cmds:
                        print(f"  {cmd}")
                    while True:
                        user_input = input(f"Please confirm whether to execute the command of {host}? (Y/N): ").strip().lower()
                        if user_input in ('y', 'n'):
                            confirm_dict[host] = user_input
                            break
                        else:
                            print("Please input 'Y' or 'N'.")
                # 再次请求，带上确认参数和 cmds_for_confirm
                result2 = await _send(client, {"query": query, "confirm": confirm_dict, "cmds_for_confirm": cmds_for_confirm})
                pretty_print_output(result2)
                # print("=" * 70)
            else:
                # 打印纯文本结果
                # print("\n" + "=" * 70)
                print(plain_text)
                # print("=" * 70)
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
        print(f"URL: {e.request.url}")
    except httpx.RequestError as e:
        print(f"❌ 请求错误: {e}")
        print(f"URL: {getattr(e.request, 'url', '未知')}")
    except asyncio.TimeoutError:
        print("❌ 请求超时，请检查网络或服务状态。")
    except json.JSONDecodeError as e:
        print(f"❌ 响应内容不是有效的 JSON: {e}")
    except Exception as e:
        print(f"❌ Unknown error: {e}")
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description='AI 助手命令行工具')
    parser.add_argument('query', nargs='?', help='查询内容')
    args = parser.parse_args()
    
    try:
        # 如果没有提供命令行参数，则使用交互式输入
        if args.query is None:
            query = input("🎯 Please input your task：")
        else:
            query = args.query

        asyncio.run(agent_quest(query))
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Exit seccess! Bye!")
        sys.exit(0)


if __name__ == '__main__':
    main()

import os, sys, json, httpx, uuid, asyncio, argparse, traceback
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:google.protobuf.runtime_version'
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')


HOST, PORT = "a6000-G5500-V6", 5003
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
    return json.loads(result["result"]["parts"][0]["text"])


async def agent_quest(query: str):
    base_url = f"http://{HOST}:{PORT}"
    try:
        async with httpx.AsyncClient(verify=False, timeout=300, proxy=None) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url,)
            final_agent_card_to_use = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)

            print(f"📤 Sending task...")
            plan = await _send(client, query)
            if plan.get("status") != "need_user_choice":
                print("⚠️ 无候选剧本:", plan)
                return

            print(f"\n{'=='*10} 候选剧本 {'=='*10}")
            for i, pb in enumerate(plan["candidates"], 1):
                print(f"{i}. {pb['name']}  -- {pb['desc']}  -- {pb['path']}")
            idxs = input("\n选择要执行的剧本序号（可多选，逗号分隔）：").strip()
            
            chosen_paths = []
            if idxs:
                for tok in idxs.split(","):
                    try:
                        chosen_paths.append(
                            (plan["candidates"][int(tok) - 1]["name"], plan["candidates"][int(tok) - 1]["path"],
                            plan["candidates"][int(tok) - 1]["host"], plan["candidates"][int(tok) - 1]["params"])
                        )
                    except Exception:
                        pass
            if not chosen_paths:
                print("🚫 未选择或所选剧本不存在，结束任务。")
                return

            # ------- Phase 2: 请求参数清单 (触发远程 cat + 解析) -------
            param_plan = await _send(client, {"chosen_paths": chosen_paths})
            if param_plan.get("status") != "need_params":
                print("⚠️ 未返回参数清单:", param_plan)
                return

            approved = []
            for pb in param_plan["playbooks"]:
                print(f"\n请填写剧本 {pb['name']} 所需传入的参数值")
                extra_vars = {}
                for prm in pb["params"]:
                    default = prm.get("default", "")
                    val = (
                        input(f"{prm['prompt']} [{default}]: ").strip() or default
                    )
                    if val != "":
                        extra_vars[prm["name"]] = val
                approved.append({"path": pb["path"], "extra_vars": extra_vars})

            # ------- Phase 3: 请求最终 shell 命令 -------
            result = await _send(client, {"param_values": approved})
            if result.get("status") != "done":
                print("❌ 未得到命令:", result)
                return

            print("\n✅ 生成命令：")
            for cmd in result["shell_cmds"]:
                print("-", cmd)
            # send each cmd to ExecAgent

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
    parser = argparse.ArgumentParser(description="A2A CLI – 选剧本/填参/执行")
    parser.add_argument("query", nargs="?", help="任务描述（留空则交互输入）")
    args = parser.parse_args()
    
    # 如果没有提供命令行参数，则使用交互式输入
    if args.query is None:
        query = input("🎯 请输入任务：")
    else:
        query = args.query

    if not query.strip():
        print("⚠️  未输入内容，已退出。")
        return

    asyncio.run(agent_quest(query))


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Exit seccess! Bye!")
        sys.exit(0)

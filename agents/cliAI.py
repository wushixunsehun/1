import sys, json, argparse, asyncio, traceback, httpx, time

HOST = "a6000-G5500-V6"
PORT = 5000
# task = "查看节点 cn79873 上【作业 20250625 的运行情况】、【slurmd 服务的运行情况】和【重启 slurmd 服务】，如果存在异常，需要分析并给出报告"
# task = "Lustre 文件系统如何通过设置精细控制 Changelog 日志的采集范围？在哪些场景下需要这样做？"
# task = "服务器 a6000 的GPU指标出现异常，进行异常情况分析"
# task = "节点 cn79873 的磁盘出现故障，进行故障分析和根因定位"
# task = "服务器 a6000 上的 rca_api 服务出现异常，制定修复策略、执行修复并验证结果"


async def agent_quest_stream(query: str):
    """流式请求处理 - 标准OpenAI格式"""
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": True  # 启用流式输出
    }

    try:
        async with httpx.AsyncClient(verify=False, timeout=3000) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        if line.startswith("data: "):
                            line = line[6:]  # 去掉 "data: " 前缀
                        
                        if line.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(line)
                            
                            # 处理标准OpenAI格式的流式数据
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    print(delta["content"], end="", flush=True)
                                    
                        except json.JSONDecodeError as e:
                            continue
                            
    except Exception as e:
        print(f"❌ HTTP请求失败: {e}")
        traceback.print_exc()


async def agent_quest(query: str):
    """非流式请求处理 - 支持前端友好格式"""
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": False  # 非流式模式
    }

    try:
        async with httpx.AsyncClient(verify=False, timeout=3000) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = json.loads(response.text)
            
            # 处理新的前端友好格式
            if isinstance(result, dict) and "type" in result:
                result_type = result.get("type")
                
                if result_type == "complete_response":
                    content = result.get("content", "")
                    print(content)
                    
                elif result_type == "error_response":
                    # 错误响应格式
                    error_info = result.get("error", {})
                    
                    error_msg = error_info.get("message", "未知错误")
                    error_type = error_info.get("type", "unknown")
                    error_code = error_info.get("code", "unknown")
                    
                    print(f"\n❌ 错误信息:")
                    print(f"   消息: {error_msg}")
                    print(f"   类型: {error_type}")
                    print(f"   代码: {error_code}")
                    
                else:
                    # 未知格式，显示原始数据
                    print(f"🔍 未知响应格式: {result_type}")
                    print("📄 原始数据:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    
            else:
                # 向后兼容：处理旧格式或其他格式
                await handle_interaction(result, query, client, headers, url)
                
    except Exception as e:
        print(f"❌ HTTP请求失败: {e}")
        traceback.print_exc()


# 通用交互处理，可扩展多种人机交互类型
async def handle_interaction(result: dict, query: str, client: httpx.AsyncClient, headers: dict, url: str):
    status = result.get("status")
    if status == "need_params_confirm":
        playbooks = result.get("playbooks", {})
        print("⚠️  需要参数确认，请按提示填写：")
        confirmed_params = {}
        for host, pb in playbooks.items():
            if pb == "无合适剧本":
                continue
            confirmed_params[host] = []
            for playbook in pb.get("playbooks", []):
                print(f"主机: {host} | 剧本: {playbook.get('name')} ({playbook.get('file')})")
                params = []
                for param in playbook.get("parameters", []):
                    val = param.get("value", "")
                    user_val = input(f"参数 {param.get('name')}，[当前 {val}]: ")
                    param["value"] = user_val if user_val.strip() else val
                    params.append(param)
                playbook["parameters"] = params
                confirmed_params[host].append(playbook)
        # 构造新的 state，补充 confirmed_params 字段
        new_state = dict(result)
        new_state["confirmed_params"] = confirmed_params
        print("📤 提交确认后的参数...")
        response2 = await client.post(url, headers=headers, json=new_state)
        response2.raise_for_status()
        result2 = json.loads(response2.text)
        await handle_interaction(result2, query, client, headers, url)
    else:
        print(result['choices'][0]['message']['content'])


def main() -> None:
    parser = argparse.ArgumentParser(description='AI 助手命令行工具 (支持前端友好格式)')
    parser.add_argument('query', nargs='?', help='查询内容')
    args = parser.parse_args()

    # 如果没有提供命令行参数，则使用交互式输入
    if args.query is None:

        query = input("🎯 请输入任务：")

    else:
        query = args.query

    if not query.strip():
        print("⚠️  未输入内容，已退出。")
        return

    asyncio.run(agent_quest_stream(query))


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 已退出，再见！")
        sys.exit(0)

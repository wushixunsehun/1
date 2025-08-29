
import sys
import termios
import tty
import yaml
from milvus_client import PlaybookMilvusClient, print_table_header, print_table_row


def show_table(items):
    if not items:
        print("无数据。")
        return
    print_table_header()
    for item in items:
        print_table_row(item)



def press_any_key():
    print("按任意键继续...", end='', flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            # 检查是否为特殊按键（如方向键、功能键等，通常以\x1b开头）
            if ch == '\x1b':
                # 读取剩余转义序列
                sys.stdin.read(2)
                continue
            else:
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print()


def main() -> None:
    client = PlaybookMilvusClient()

    while True:
        print("\n请选择一个操作：")
        print("┌─────────────────────────────────────────────────────┐")
        print("│ 1. 列出所有剧本         │ 2. 根据文件名检索剧本信息 │")
        print("│ 3. 根据描述检索剧本信息 │ 4. 根据ID检索剧本信息     │")
        print("│ 5. 根据ID获取剧本内容   │ 6. 根据文件名获取剧本内容 │")
        print("│ 7. 新增剧本             │ 8. 根据ID更新剧本         │")
        print("│ 9. 根据ID删除剧本       │ 10. 根据文件名删除剧本    │")
        print("│ 0. 退出                                             │")
        print("└─────────────────────────────────────────────────────┘")

        choice = input("请输入操作编号：")

        try:

            if choice == "1":
                results = client.list_all()
                print("\n当前剧本列表：")
                show_table(results)
                press_any_key()

            elif choice == "2":
                playbook_name = input("请输入剧本名称：")
                result = client.search_playbooks_by_file(playbook_name)
                # print(f"\n名称为 [{playbook_name}] 的剧本：")
                show_table([result])
                press_any_key()

            elif choice == "3":
                query_text = input("请输入描述关键词：")
                top_k = int(input("请输入返回的剧本数量（默认 3）：") or 3)
                results = client.search_playbooks_by_desc(query_text, top_k)
                print(f"\n与描述 [{query_text}] 相似的剧本：")
                show_table(results)
                press_any_key()

            elif choice == "4":
                playbook_id = int(input("请输入剧本 ID："))
                result = client.get_playbook_by_id(playbook_id)
                # print(f"\nID 为 [{playbook_id}] 的剧本信息：")
                show_table([result])
                press_any_key()

            elif choice == "5":
                playbook_id = int(input("请输入剧本 ID："))
                content = client.get_playbook_content_by_id(playbook_id)
                print(f"\n{content}")
                press_any_key()

            elif choice == "6":
                playbook_file = input("请输入剧本文件名：")
                content = client.get_playbook_content_by_file(playbook_file)
                print(f"\n{content}")
                press_any_key()

            elif choice == "7":
                file_path = input("请输入剧本文件的绝对路径：")
                client.add_playbook(file_path)
                print(f"剧本 [{file_path.split('/')[-1]}] 添加成功。")
                press_any_key()

            elif choice == "8":
                playbook_id = int(input("请输入剧本 ID："))
                result = client.get_playbook_by_id(playbook_id)
                show_table([result])

                new_data = {}
                print("请输入需要更新的字段（留空跳过）：")


                file = input("新文件名：")
                if file.strip():
                    new_data["file"] = file

                name = input("新名称：")
                if name.strip():
                    new_data["name"] = name

                description = input("新描述：")
                if description.strip():
                    new_data["description"] = description

                content = input("新内容：")
                if content.strip():
                    raw = content.strip()
                    # 如果是一行带\n的字符串，先还原为多行
                    if "\\n" in raw:
                        raw = raw.replace("\\n", "\n")
                    lines = raw.splitlines()
                    # 自动转换为 YAML block string
                    if lines and not lines[0].startswith('|'):
                        block_content = '|'
                        for line in lines:
                            block_content += '\n  ' + line
                        new_data["content"] = block_content
                    else:
                        new_data["content"] = raw

                parameters = input("新参数：")
                """
                单参数： - {name: jobid, type: str}
                多参数：[{"name": "jobid", "type": "str"}, {"name": "force", "type": "bool"}]
                """
                if parameters.strip():
                    try:
                        # 支持直接输入YAML字符串或简单列表
                        params_obj = yaml.safe_load(parameters)
                        new_data["parameters"] = yaml.dump(params_obj, allow_unicode=True)
                    except Exception:
                        # 输入非YAML格式时直接存字符串
                        raise ValueError("参数格式错误，请输入有效的 YAML 或 JSON 格式。")

                user_id = input("新用户 ID：")
                if user_id.strip():
                    new_data["user_id"] = user_id

                client.update_playbook(playbook_id, new_data)
                press_any_key()

            elif choice == "9":
                playbook_id = int(input("请输入剧本 ID："))
                client.delete_playbook_by_id(playbook_id)
                print("剧本删除成功。")
                press_any_key()

            elif choice == "10":
                file_name = input("请输入剧本文件名：")
                client.delete_playbook_by_file(file_name)
                print("剧本删除成功。")
                press_any_key()

            elif choice == "0":
                print("退出程序。")
                break

            else:
                print("无效的选择，请重新输入。")

        except Exception as e:
            print(f"操作失败：{e}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n退出程序。")
        sys.exit(0)

import os, re, yaml
from tqdm import tqdm
from milvus_client import PlaybookMilvusClient


id_width = 10
file_width = 25
name_width = 20
desc_width = 60
params_width = 25


def estimate_width(text: str) -> int:
    """估算终端显示宽度（中文算2，英文算1）"""
    width = 0
    for c in text:
        width += 2 if ord(c) > 127 else 1
    return width


def pad_display(text: str, width: int) -> str:
    """按估算宽度对齐，右侧填空格"""
    padding = width - estimate_width(text)
    return text + ' ' * max(padding, 0)


def print_table_header():
    print(
        f"\n{pad_display('ID', id_width)} {pad_display('文件名', file_width)} {pad_display('剧本名称', name_width)} {pad_display('描述', desc_width)} {pad_display('所需参数', params_width)}"
    )
    print("-" * 140)


def wrap_text(text: str, width: int) -> list[str]:
    """按照估算宽度自动换行"""
    lines, current, cur_width = [], '', 0
    for c in text:
        w = 2 if ord(c) > 127 else 1
        if cur_width + w > width:
            lines.append(current)
            current, cur_width = '', 0
        current += c
        cur_width += w
    if current:
        lines.append(current)
    return lines


def print_table_row(item: dict):
    _id_lines = wrap_text(str(item["id"]), id_width)
    file_lines = wrap_text(str(item["file"]), file_width)
    name_lines = wrap_text(str(item["name"]), name_width)
    desc_lines = wrap_text(str(item["description"]), desc_width)

    # 处理参数列，结构化美化输出
    try:
        params_raw = item.get("parameters", "")
        params_list = []
        if isinstance(params_raw, str) and params_raw.strip():
            try:
                params_yaml = yaml.safe_load(params_raw)
                if isinstance(params_yaml, list):
                    for p in params_yaml:
                        if isinstance(p, dict):
                            pname = str(p.get('name', ''))
                            ptype = str(p.get('type', ''))
                            params_list.append(f"{pname:<10} {ptype}")
                        else:
                            params_list.append(str(p))
                else:
                    params_list.append(str(params_yaml))
            except Exception:
                # fallback: 尝试用正则解析
                for m in re.finditer(r'- name: (\S+)\s+type: (\S+)', params_raw, re.MULTILINE):
                    params_list.append(f"{m.group(1):<10} {m.group(2)}")
                if not params_list:
                    params_list = params_raw.splitlines()
        elif isinstance(params_raw, list):
            for p in params_raw:
                if isinstance(p, dict):
                    pname = str(p.get('name', ''))
                    ptype = str(p.get('type', ''))
                    params_list.append(f"{pname:<10} {ptype}")
                else:
                    params_list.append(str(p))
        else:
            params_list = [str(params_raw)] if params_raw else []
        if not params_list:
            params_list = ['']
        params_lines = []
        for pl in params_list:
            params_lines.extend(wrap_text(pl, params_width))
    except Exception:
        params_lines = [str(item.get("parameters", ""))]

    max_lines = max(len(_id_lines), len(file_lines), len(name_lines), len(desc_lines), len(params_lines))

    for i in range(max_lines):
        id_part = pad_display(_id_lines[i] if i < len(_id_lines) else '', id_width)
        file_part = pad_display(file_lines[i] if i < len(file_lines) else '', file_width)
        name_part = pad_display(name_lines[i] if i < len(name_lines) else '', name_width)
        desc_part = pad_display(desc_lines[i] if i < len(desc_lines) else '', desc_width)
        params_part = pad_display(params_lines[i] if i < len(params_lines) else '', params_width)
        print(f"{id_part} {file_part} {name_part} {desc_part} {params_part}")
    print("-" * 140)


def main():
    current_dir = '/home/tanxh/mas/agents/thsre'
    m_client = PlaybookMilvusClient()
    yml_files = [f for f in os.listdir(current_dir) if f.endswith('.yml')]

    # res = m_client.list_all()
    # print_table_header()
    # for item in res:
    #     print_table_row(item)

    # for file_name in tqdm(yml_files):
    #     file_path = os.path.join(current_dir, file_name)

    #     try:
    #         m_client.add_playbook(file_path)
    #         # pass
    #     except Exception as e:
    #         print(f"处理 {file_name} 时出错: {e}")
    
    # m_client.search_playbooks_by_name('恢复作业和分区状态')
    # m_client.get_playbook_by_id(9059384074831552121)
    # data={
    #     "name": '调整队列独占模式',
    #     "content": ''
    # }

    # m_client.update_playbook(6934764437346756659, data)
    # m_client.delete_playbook_by_id(6934764437346756659)
    # m_client.delete_playbook_by_file('show_node.yml')
    # m_client.get_playbook_content_by_id(4601227412437227906)
    res = m_client.list_all()
    print_table_header()
    for item in res:
        print_table_row(item)


if __name__ == "__main__":
    main()

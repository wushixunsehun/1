import re, os, sys, yaml
sys.path.append('./')
from pathlib import Path
import hashlib
from pymilvus import DataType, MilvusClient
from agentsAPI import query_llm, query_embedding, strip_think


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


STRATEGY_PROMPT_TEMPLATE = """你是一名资深 DevOps 工程师，熟悉 Ansible 语言规范与最佳实践。以下是一个 Ansible 剧本的内容：

---
{playbook_content}

请你从该剧本中提取结构化信息，**仅输出以下 YAML 字段结构，字段顺序必须完全一致**：

- name: 中文剧本名（对剧本整体功能进行高度概括，避免照抄 task 名或变量名）
- description: 中文描述，说明该剧本的实际作用或用途
- content: 原始剧本内容，以 YAML 的 block 字符串形式嵌入（| 开头）
- parameters: 外部输入参数列表，仅包括需要通过 -e 或变量机制传入的字段
  - name: 参数名
    type: 参数类型（如 int、str、bool 等，需根据变量使用方式和上下文判断）
- user_id: ""（留空）

### 严格要求如下：

1. 输出结果必须是合法 YAML 格式，**不能有任何 Markdown 格式（如 
yaml）或注释文字**；
2. `content` 字段必须使用 `|` 语法嵌入完整剧本内容，每行前保持两个空格缩进，禁止展开为 YAML 列表或 dict，否则将导致解析错误；
3. 所有字段的顺序必须严格为以下顺序，字段名前不能缩进：
   - name  
   - description  
   - content  
   - parameters  
   - user_id
4. 若剧本不包含任何外部参数，`parameters` 字段仍必须保留，并设置为 `[]`；
5. 整体输出中不得额外添加说明性语言、解释性文字或 JSON 格式结构，仅输出 YAML 数据本体。

请你严格遵守格式和字段顺序，确保输出结果可被 YAML 解析器直接读取。
"""

class PlaybookMilvusClient:
    def __init__(self,
                uri: str = "http://a6000-G5500-V6:19530",
                token: str = "tanxh:pwd@123",
                db_name: str = "Ansible_playbook",
                collection_name: str = "playbooks",
                embedding_dim: int = 768
    ):
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = self._connect()


    def stable_id(self, text: str, digits: int = 8) -> int:
        hash_digest = hashlib.md5(text.encode()).hexdigest()
        id_int = int(hash_digest, 16)
        max_val = 10 ** digits
        return id_int % max_val


    def _connect(self):
        # 初始化 Milvus 客户端
        client = MilvusClient(uri=self.uri, token=self.token, db_name=self.db_name)
        # client.create_database(db_name=DB_NAME)

        # 创建集合（如不存在）
        # client.drop_collection(self.collection_name)    # 删除集合
        # client.delete(collection_name=self.collection_name, filter="id >= 0")    # 删除集合里的数据

        if not client.has_collection(self.collection_name):
            schema = client.create_schema(auto_id=False)
            schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='Playbook ID')
            schema.add_field(field_name='user_id', datatype=DataType.VARCHAR, max_length=512, description='User ID Text')
            schema.add_field(field_name='file', datatype=DataType.VARCHAR, max_length=256, description='Playbook File Text')
            schema.add_field(field_name='name', datatype=DataType.VARCHAR, max_length=512, description='Playbook Name Text')
            schema.add_field(field_name='description', datatype=DataType.VARCHAR, max_length=1024, description='Playbook Desc Text')
            schema.add_field(field_name='description_embedding', datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim, description='Playbook description Vector')
            schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=4096, description='Playbook Content Text')
            schema.add_field(field_name='parameters', datatype=DataType.VARCHAR, max_length=1024, description='Playbook Parameters Text')

            index_params = client.prepare_index_params()
            index_params.add_index(field_name='description_embedding', index_type='AUTOINDEX', metric_type='IP')
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )

        # client.load_collection(COLLECTION_NAME)
        return client


    def _get_embedding(self, text: str) -> list[float]:
        return query_embedding([text])


    def _parse_playbook(self, playbook_content: str) -> dict:
        enable_thinking = True
        prompt = STRATEGY_PROMPT_TEMPLATE.format(playbook_content=playbook_content)

        enable_thinking = True
        if enable_thinking:
            response = query_llm(prompt, enable_thinking=enable_thinking)
            response = strip_think(response).strip()
        else:
            response = query_llm(prompt)

        response = re.sub(r"^```yaml\s*|\s*```$", "", response, flags=re.IGNORECASE)
        parsed = yaml.safe_load(response)

        return parsed


    # API 1：列出所有剧本（返回 ID、名称、描述）
    def list_all(self):
        results = self.client.query(
            collection_name=self.collection_name,
            output_fields=["id", "file", "name", "description", "content", "parameters"],
            filter="",
            limit=500,
        )

        if not results:
            print("知识库为空，无剧本数据。")
            return

        return results


    # API 2：模糊名称检索 / 精确 ID 检索
    def search_playbooks_by_desc(self, query_text: str, top_k=3):
        if not query_text:
            raise ValueError("查询文本不能为空。")
        
        # print(f"\n正在检索相似剧本...\n")
        embedding = self._get_embedding(query_text)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="description_embedding",
            output_fields=["id", "file", "name", "description", "content", "parameters"],
            limit=top_k
        )

        if not results:
            raise ValueError("未找到匹配结果。")

        return results[0]


    def search_playbooks_by_file(self, file: str):
        if not file:
            raise ValueError("剧本文件名不能为空。")

        # 修正表达式，字符串需加引号
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f"file == '{file}'",
            output_fields=["id", "file", "name", "description", "content", "parameters"]
        )

        if not results:
            raise ValueError(f"未找到剧本文件 {file}。")

        return results[0]


    def get_playbook_by_id(self, playbook_id: int):
        if not playbook_id:
            raise ValueError("剧本 ID 不能为空。")
        if not isinstance(playbook_id, int):
            raise ValueError("剧本 ID 必须是整数。")
        if not playbook_id > 0:
            raise ValueError("剧本 ID 必须大于 0。")
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'id == {playbook_id}',
            output_fields=["id", "file", "name", "description", "content", "parameters"]
        )

        if not results:
            raise ValueError(f"未找到 ID 为 {playbook_id} 的剧本。")

        return results[0]


    def get_playbook_content_by_id(self, playbook_id: int):
        if not playbook_id:
            raise ValueError("剧本 ID 不能为空。")
        if not isinstance(playbook_id, int):
            raise ValueError("剧本 ID 必须是整数。")
        if not playbook_id > 0:
            raise ValueError("剧本 ID 必须大于 0。")
        
        results = self.client.query(
            collection_name=self.collection_name,
            filter=f'id == {playbook_id}',
            output_fields=["content"]
        )

        if not results:
            raise ValueError(f"未找到 ID 为 {playbook_id} 的剧本。")

        return results[0]['content']


    def get_playbook_content_by_file(self, file: str):
        if not file:
            raise ValueError("剧本文件名不能为空。")

        results = self.client.query(
            collection_name=self.collection_name,
            filter=f"file == '{file}'",
            output_fields=["content"]
        )

        if not results:
            raise ValueError(f"未找到名为 {file} 的剧本。")

        return results[0]['content']


    # API 3：新增剧本（通过绝对文件路径）
    def add_playbook(self, file_path: str):
        if not os.path.exists(file_path):
            raise ValueError(f"路径 {file_path} 不存在。")
        
        # --- 先判断是否存在剧本文件数据 ---
        file_name = os.path.basename(file_path)
        existing = self.client.query(
            collection_name=self.collection_name,
            filter=f"file == '{file_name}'",
            output_fields=["id"]
        )
        if existing:
            raise ValueError(f"剧本 {file_path} 已存在。")

        # --- 否则，读取剧本内容并解析 ---
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        desc = self._parse_playbook(content)
        new_id = self.stable_id(desc["name"])

        # --- 插入新剧本数据 ---
        self.client.insert(
            collection_name=self.collection_name,
            data=[{
                "id": new_id,
                "user_id": desc.get("user_id", ""),
                "file": file_name,
                "name": desc["name"],
                "description": desc["description"],
                "description_embedding": self._get_embedding(f'{desc["name"]}\n{file_name}\n{desc["description"]}\n{desc["content"]}\n{desc.get("parameters", [])}'),
                "content": desc["content"],
                "parameters": yaml.dump(desc.get("parameters", []), allow_unicode=True)
            }]
        )
        # print("\n成功添加剧本：")
        # print_table_header()
        # print_table_row({"id": new_id, "file": file_name, "name": desc["name"], "description": desc["description"]})


    # API 4：更新剧本（根据 ID）
    def update_playbook(self, playbook_id: int, new_data: dict):
        existing = self.client.query(
            collection_name=self.collection_name,
            filter=f'id == {playbook_id}',
            output_fields=["id", "file", "name", "description", "description_embedding", "content", "parameters", "user_id"]
        )

        if not existing:
            raise ValueError(f"剧本 ID {playbook_id} 不存在")

        updated = existing[0].copy()
        updated.update(new_data)

        if "description" in new_data and new_data["description"] != existing[0]["description"]:
            updated["description_embedding"] = self._get_embedding(f'{updated["name"]}\n{updated["file"]}\n{new_data["description"]}\n{updated["content"]}\n{updated.get("parameters", [])}')


        self.client.delete(self.collection_name, filter=f"id == {playbook_id}")
        result = self.client.insert(self.collection_name, data=[updated])
        new_id = result['ids'][0]

        # print("\n剧本更新成功！")
        # print_table_header()
        # print_table_row({"id": new_id, "file": updated["file"], "name": updated["name"], "description": updated["description"]})

        if new_data:
            print("\n📌 更新字段：")
            for k, v in new_data.items():
                if k != "description_embedding":
                    print(f"- {k}: {v}")


    # API 5：删除剧本（根据 ID）
    def delete_playbook_by_id(self, playbook_id: int):
        if not playbook_id:
            raise ValueError("剧本 ID 不能为空。")
        if not isinstance(playbook_id, int):
            raise ValueError("剧本 ID 必须是整数。")
        if not playbook_id > 0:
            raise ValueError("剧本 ID 必须大于 0。")
        
        existing = self.client.query(
            collection_name=self.collection_name,
            filter=f'id == {playbook_id}',
            output_fields=["id", "file", "name", "description"]
        )

        if not existing:
            raise ValueError(f"未找到 ID 为 {playbook_id} 的剧本，无法删除。")

        self.client.delete(self.collection_name, filter=f"id == {playbook_id}")


    # API 6：删除剧本（根据文件名）
    def delete_playbook_by_file(self, file: str):
        if not file:
            raise ValueError("剧本文件名不能为空。")
        
        existing = self.client.query(
            collection_name=self.collection_name,
            filter=f"file == '{file}'",
            output_fields=["id", "file", "name", "description"]
        )

        if not existing:
            raise ValueError(f"未找到名为 {file} 的剧本，无法删除。")

        self.client.delete(self.collection_name, filter=f"file == '{file}'")

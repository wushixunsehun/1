import hashlib
import os
from tqdm import tqdm
from pymilvus import MilvusClient, DataType
from client import query_llm


class Milvus:
    def __init__(self, config: dict):
        # 指定一个存储数据的文件
        self.client = MilvusClient(uri=config['uri'], token=config['token'], db_name=config['db'])

        self.summary_collection_name = config['summary_collection_name']
        self.content_collection_name = config['content_collection_name']
        self.s2c_collection_name = config['s2c_collection_name']


        self.doc_emb_dim = config['doc_emb_dim']
        self.embedding_model = config['embedding_model']

        # self.db_name = 'chunk_500'
        # self.client.create_database(db_name=self.db_name)
        # self.client.use_database(db_name=self.db_name)

        # self.drop_collection(self.summary_collection_name)
        # self.drop_collection(self.content_collection_name)
        # self.drop_collection(self.s2c_collection_name)

        # self.create_collectionsv2()

        # 验证 token 有效性
        # try:
        #     self.client.list_collections()
        # except Exception as e:
        #     raise RuntimeError(f"Milvus token 无效或已过期: {e}")


    def list_collections(self):
        """列出当前连接的数据库中所有集合的名称列表"""
        print(self.client.list_collections())


    def drop_collection(self, collection_name):
        """删除指定的 collection_name"""
        self.client.drop_collection(collection_name = collection_name)
        print(f'Drop collection [{collection_name}] success!')


    def describe_collection(self, collection_name):
        """获取特定 collection_name 的详细信息"""
        collection_desc = self.client.describe_collection(
            collection_name = collection_name
        )
        print(collection_desc)


    def create_collectionsv2(self):
        """创建集合"""
        # 创建摘要集合
        summary_schema = self.client.create_schema(auto_id=False)
        summary_schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='Summary ID')
        summary_schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=self.doc_emb_dim, description='Summary Vector')
        summary_schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=8192, description='Summary Text')
        summary_schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=128, description='Summary Source')
        summary_schema.add_field(field_name='doc_name', datatype=DataType.VARCHAR, max_length=512, description='Document Name')
        summary_schema.add_field(field_name='doc_path', datatype=DataType.VARCHAR, max_length=512, description='DOcument Path')

        summary_index_params = self.client.prepare_index_params()

        summary_index_params.add_index(field_name = 'vector', index_type = 'AUTOINDEX', metric_type = 'IP')
        summary_index_params.add_index(field_name = 'source', index_type = 'INVERTED')

        if not self.client.has_collection(self.summary_collection_name):
            self.client.create_collection(
                collection_name = self.summary_collection_name,
                schema = summary_schema,
                index_params = summary_index_params
            )


        # 创建内容集合
        content_schema = self.client.create_schema(auto_id=False)
        content_schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='Content ID')
        content_schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=self.doc_emb_dim, description='Content Vector')
        content_schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=8192, description='Content Text')
        content_schema.add_field(field_name='doc_name', datatype=DataType.VARCHAR, max_length=512, description='Document Name')
        content_schema.add_field(field_name='doc_path', datatype=DataType.VARCHAR, max_length=512, description='DOcument Path')

        content_index_params = self.client.prepare_index_params()

        content_index_params.add_index(field_name='vector', index_type='AUTOINDEX', metric_type='IP')

        if not self.client.has_collection(self.content_collection_name):
            self.client.create_collection(
                collection_name = self.content_collection_name,
                schema = content_schema,
                index_params = content_index_params
            )


        # 创建摘要与内容关联的集合
        s2c_schema = self.client.create_schema(auto_id=True)
        s2c_schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='ID')
        s2c_schema.add_field(field_name='s_id', datatype=DataType.INT64, description='Summary ID')
        s2c_schema.add_field(field_name='c_id', datatype=DataType.INT64, description='Content ID')
        s2c_schema.add_field(field_name='s_vector', datatype=DataType.FLOAT_VECTOR, dim=self.doc_emb_dim, description='Summary Vector')
        s2c_schema.add_field(field_name='c_vector', datatype=DataType.FLOAT_VECTOR, dim=self.doc_emb_dim, description='Content Vector')

        s2c_index_params = self.client.prepare_index_params()

        s2c_index_params.add_index(field_name='s_vector', index_type='AUTOINDEX', metric_type='IP')
        s2c_index_params.add_index(field_name='c_vector', index_type='AUTOINDEX', metric_type='IP')

        if not self.client.has_collection(self.s2c_collection_name):
            self.client.create_collection(
                collection_name = self.s2c_collection_name,
                schema = s2c_schema,
                index_params = s2c_index_params
            )


        # 装载集合，这是在集合中进行相似性搜索和查询的前提
        # self.client.load_collection(self.summary_collection_name)
        # self.client.load_collection(self.content_collection_name)
        # self.client.load_collection(self.s2c_collection_name)


    def stable_id(self, name: str, prefix: str = None) -> int:
        # return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
        raw = f"{prefix}:{name}" if prefix else name

        # 使用 md5 生成 128bit 哈希，取前16位十六进制（64位）
        hash_digest = hashlib.md5(raw.encode()).hexdigest()[:16]

        # 转换为 64bit 整数，限制最大值为 2^63 - 1（INT64 正数上限）
        id_64bit = int(hash_digest, 16) & 0x7FFFFFFFFFFFFFFF  # 强制确保为正整数

        return id_64bit


    def docs_embedding(self, docs):
        # 准备数据
        summary_data = []
        content_data = []
        s2c_data = []

        PROMPT_TEMPLATE = '''你是一名专业的文本分析专家，擅长根据文本内容判断其来源类型。
        可能的来源类型有：system、lustre、promql、slurm、handbook、ticket、general。

        要求：
        - 根据以下文本内容进行分析，仅返回这段文本的内容最符合哪种来源类型，不引入额外的主观假设和解释。
        - 确保判断结果清晰明确，便于后续操作。

        请直接输出以下文本的来源类型：{text}
        '''

        for doc_abs_path, doc_data in tqdm(docs.items(), desc='文档嵌入'):
            doc_name = os.path.basename(doc_abs_path)
            for docs_data in doc_data:
                summary_text = docs_data['summary']
                chunks = docs_data['chunks']

                prompt = PROMPT_TEMPLATE.format(text=summary_text)
                source = query_llm(prompt)

                # 使用 encode_documents 对文档内容进行嵌入
                summary_id = self.stable_id(summary_text, source)
                summary_vector = self.embedding_model.encode(summary_text, normalize_embeddings=True)

                summary_data.append({
                    'id': summary_id,
                    'vector': summary_vector,
                    'text': summary_text,
                    'source': source,
                    'doc_name': doc_name,
                    'doc_path': doc_abs_path
                })

                for i, chunk in enumerate(chunks):
                    content_id = self.stable_id(f"{summary_text}_chunk_{i}")
                    content_vector = self.embedding_model.encode(chunk, normalize_embeddings=True)

                    content_data.append({
                        'id': content_id,
                        'vector': content_vector,
                        'text': chunk,
                        'doc_name': doc_name,
                        'doc_path': doc_abs_path
                    })

                    s2c_data.append({
                        's_id': summary_id,
                        'c_id': content_id,
                        's_vector': summary_vector,
                        'c_vector': content_vector
                    })

        return summary_data, content_data, s2c_data


    def insert_data(self, summary_data, content_data, s2c_data):
        """按各自长度独立插入摘要、内容、映射数据"""
        def insert_in_batches(data, collection_name):
            for i in range(0, len(data), 500):
                batch = data[i:i + 500]
                self.client.insert(collection_name=collection_name, data=batch)

        insert_in_batches(summary_data, self.summary_collection_name)
        insert_in_batches(content_data, self.content_collection_name)
        insert_in_batches(s2c_data, self.s2c_collection_name)

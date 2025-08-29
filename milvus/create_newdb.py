from pymilvus import MilvusClient, DataType

def create_chunk_test_db(config):
    # 创建数据库
    client = MilvusClient(uri=config['uri'], token=config['token'])
    db_name = "chunk_test_v2"
    if db_name not in client.list_databases():
        client.create_database(db_name=db_name)
    client.using_database(db_name=db_name)

    # client.drop_collection(config['summary_collection_name'])
    # client.drop_collection(config['content_collection_name'])
    # client.drop_collection(config['s2c_collection_name'])

    # 创建摘要集合
    summary_schema = client.create_schema(auto_id=False)
    summary_schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='Summary ID')
    summary_schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=config['doc_emb_dim'], description='Summary Vector')
    summary_schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=8192, description='Summary Text')
    summary_schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=128, description='Summary Source')
    summary_schema.add_field(field_name='doc_name', datatype=DataType.VARCHAR, max_length=512, description='Document Name')
    summary_schema.add_field(field_name='doc_path', datatype=DataType.VARCHAR, max_length=512, description='Document Path')

    summary_index_params = client.prepare_index_params()
    summary_index_params.add_index(field_name='vector', index_type='AUTOINDEX', metric_type='L2')
    summary_index_params.add_index(field_name='source', index_type='INVERTED')

    if not client.has_collection(config['summary_collection_name']):
        client.create_collection(
            collection_name=config['summary_collection_name'],
            schema=summary_schema,
            index_params=summary_index_params
        )

    # 创建内容集合
    content_schema = client.create_schema(auto_id=False)
    content_schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='Content ID')
    content_schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=config['doc_emb_dim'], description='Content Vector')
    content_schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=8192, description='Content Text')
    content_schema.add_field(field_name='doc_name', datatype=DataType.VARCHAR, max_length=512, description='Document Name')
    content_schema.add_field(field_name='doc_path', datatype=DataType.VARCHAR, max_length=512, description='Document Path')

    content_index_params = client.prepare_index_params()
    content_index_params.add_index(field_name='vector', index_type='AUTOINDEX', metric_type='L2')

    if not client.has_collection(config['content_collection_name']):
        client.create_collection(
            collection_name=config['content_collection_name'],
            schema=content_schema,
            index_params=content_index_params
        )

    # 创建摘要与内容关联集合
    s2c_schema = client.create_schema(auto_id=True)
    s2c_schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='ID')
    s2c_schema.add_field(field_name='s_id', datatype=DataType.INT64, description='Summary ID')
    s2c_schema.add_field(field_name='c_id', datatype=DataType.INT64, description='Content ID')
    s2c_schema.add_field(field_name='s_vector', datatype=DataType.FLOAT_VECTOR, dim=config['doc_emb_dim'], description='Summary Vector')
    s2c_schema.add_field(field_name='c_vector', datatype=DataType.FLOAT_VECTOR, dim=config['doc_emb_dim'], description='Content Vector')

    s2c_index_params = client.prepare_index_params()
    s2c_index_params.add_index(field_name='s_vector', index_type='AUTOINDEX', metric_type='L2')
    s2c_index_params.add_index(field_name='c_vector', index_type='AUTOINDEX', metric_type='L2')

    if not client.has_collection(config['s2c_collection_name']):
        client.create_collection(
            collection_name=config['s2c_collection_name'],
            schema=s2c_schema,
            index_params=s2c_index_params
        )

if __name__ == "__main__":
    config = {
        "uri": "http://localhost:19530",  # 替换为你的 Milvus 地址
        "token": "root:lhltxh971012",                      # 如有需要填写 token
        "doc_emb_dim": 768,               # 替换为你的 embedding 维度
        "summary_collection_name": "summary",
        "content_collection_name": "content",
        "s2c_collection_name": "s2c"
    }
    create_chunk_test_db(config)
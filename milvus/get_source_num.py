from pymilvus import connections, Collection
from collections import defaultdict
import yaml
import time

def get_source_document_counts(milvus_conf: dict):
    """
    获取每个 source 的文档数量
    :param collection_name: Milvus 集合名称
    :return: 每个 source 的文档数量字典
    """
    # 连接到 Milvus
    connections.connect(alias="default", uri=milvus_conf['uri'], token=milvus_conf['token'], db_name=milvus_conf['db'])

    # 获取集合
    collection = Collection(milvus_conf['summary_collection_name'])

    # 假设集合中有一个字段名为 "source"
    source_field = "source"

    # 获取所有 source 的值
    expr = f"{source_field} != ''"  # 过滤掉空的 source
    results = collection.query(expr=expr, output_fields=[source_field])

    # 统计每个 source 的文档数量
    source_counts = defaultdict(int)
    for result in results:
        source = result[source_field]
        source_counts[source] += 1

    # 断开连接
    connections.disconnect(alias="default")

    return len(results), source_counts


def main() -> None:
    config_path = 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    milvus_conf = config['milvus']
    docs_len, source_counts = get_source_document_counts(milvus_conf)

    print(f"# chunk: {docs_len}, # source: {len(source_counts)}, # chunk in each source:")
    for source, count in source_counts.items():
        print(f"source: {source}, # chunk: {count}")

if __name__ == '__main__':
    start_main_time = time.perf_counter()
    main()
    end_main_time = time.perf_counter()
    print(f'Run time: {end_main_time - start_main_time:.4f}s')

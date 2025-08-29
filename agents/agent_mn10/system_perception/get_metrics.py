import json
import time
import yaml
import hashlib
import requests
from tqdm import tqdm
from pathlib import Path
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


def get_local_model_path(model_name: str) -> str:
    """
    解析模型名称为本地地址
    """
    safe_name = model_name.replace("/", "--")
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hub_dir / f"models--{safe_name}" / "snapshots"
    
    # 默认只取第一个 snapshot 子目录（一般只有一个）
    snapshot_dirs = list(model_dir.glob("*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for model: {model_name}")
    
    return str(snapshot_dirs[0])


# 导入配置文件
config_path = 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

milvus_conf = config['milvus']
collection_conf = config['collections']


# 连接 milvus，使用指定数据库和嵌入模型
client = MilvusClient(uri=milvus_conf['uri'], token=milvus_conf['token'], db_name=milvus_conf['db'])

summary_collection = collection_conf['summary']
content_collection = collection_conf['content']
s2c_collection = collection_conf['summary2content']

model_name = config['rag']['embedding_model']
embedding_model_path = get_local_model_path(model_name)
embedding_model = SentenceTransformer(
    embedding_model_path,
    device = config['rag']['device'],
    trust_remote_code = True,
    local_files_only = True
)


def get_prometheus_metrics(prometheus_url):
    """
    访问监控地址上的 prometheus 数据库，获取所有指标及其详细信息，返回字典
    """
    try:
        proxies = {
            "http": "",
            "https": "",
        }

        query_url = f"{prometheus_url}/api/v1/metadata"
        response = requests.get(query_url, timeout=1000, proxies=proxies)
        response.raise_for_status()
        metadata = response.json()

        if metadata.get("status") != "success":
            raise Exception("Failed to fetch metrics metadata from Prometheus")

        metrics = {}
        for metric_name, details in metadata.get("data", {}).items():
            if details:
                metrics[metric_name] = {
                    "type": details[0].get("type", "None"),
                    "help": details[0].get("help", "No description available")
                }

        return metrics

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Prometheus: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_metrics(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def merge(metric_dict: dict, all_metrics: dict):
    """
    合并新旧指标字典
    """
    for name, details in metric_dict.items():
        if name not in all_metrics:
            all_metrics[name] = details
        else:
            continue


def stable_id(name: str, prefix: str = None) -> int:
    """
    使用 128bit 哈希为数据条目生成 id
    """
    raw = f"{prefix}:{name}" if prefix else name

    # 使用 md5 生成 128bit 哈希，取前16位十六进制（64位）
    hash_digest = hashlib.md5(raw.encode()).hexdigest()[:16]

    # 转换为 64bit 整数，限制最大值为 2^63 - 1（INT64 正数上限）
    id_64bit = int(hash_digest, 16) & 0x7FFFFFFFFFFFFFFF  # 强制确保为正整数

    return id_64bit


def get_all_ids(collection_name: str):
    """
    查询所有 id
    """
    results = client.query(
        collection_name = collection_name,
        expr = "",
        output_fields = ["id"]
    )
    return set(item['id'] for item in results)


def update_vector_db(all_metrics: dict):
    """
    将指标数据嵌入并更新进知识库
    """
    summary_data = []
    content_data = []
    s2c_data = []
    source = 'metric'

    new_ids = set()
    for metric_name, metric_desc in tqdm(all_metrics.items(), desc='指标嵌入'):
        text = f'{metric_name}: {metric_desc}'

        # 使用 encode_documents 对文档内容进行嵌入
        id = stable_id(text, source)
        new_ids.add(id)
        vector = embedding_model.encode(text, normalize_embeddings=True)

        summary_data.append({
            'id': id,
            'vector': vector,
            'text': text,
            'source': source,
            'doc_name': '',
            'doc_path': ''
        })

        content_data.append({
            'id': id,
            'vector': vector,
            'text': text,
            'doc_name': '',
            'doc_path': ''
        })

        s2c_data.append({
            's_id': id,
            'c_id': id,
            's_vector': vector,
            'c_vector': vector
        })


    # 读取当前已存在的 ID
    existing_ids = get_all_ids(summary_collection)
    obsolete_ids = existing_ids - new_ids
    print(f"将删除 {len(obsolete_ids)} 条旧指标...")


    # 删除旧的 ID 数据
    if obsolete_ids:
        id_list = list(obsolete_ids)
        for collection in [summary_collection, content_collection, s2c_collection]:
            if collection == s2c_collection:
                expr = f"s_id in {id_list}"
            else:
                expr = f"id in {id_list}"

            client.delete(collection, expr=expr)


    # 分批次插入数据
    def insert_in_batches(data, collection_name):
        for i in range(0, len(data), 500):
            batch = data[i:i + 500]
            client.upsert(collection_name=collection_name, data=batch)

    insert_in_batches(summary_data, summary_collection)
    insert_in_batches(content_data, content_collection)
    insert_in_batches(s2c_data, s2c_collection)

    print("指标向量库已更新完毕。")


def main() -> None:
    machines = ['ion210', 'ion211', 'ion212']    # 后期可设定为外部传参
    metric_file_path = 'agent-mn10/prom_metrics/merged_metrics.json'
    all_metrics = read_metrics(metric_file_path)    # 读取已有的指标

    for machine in machines:
        # 遍历服务器名称，获取指标名称及其描述并返回字典
        prometheus_url = f"http://{machine}:10900"
        output_file = f"agent-mn10/prom_metrics/metrics_{machine}.json"
        new_metrics = get_prometheus_metrics(prometheus_url)

        # 保存文件
        with open(output_file, "w") as json_file:
            json.dump(new_metrics, json_file, indent=4)
        print(f"Metrics of machine【{machine}】 successfully saved to 【{output_file}】")

        # 更新指标名称及内容
        merge(new_metrics, all_metrics)

    # 获取完所有服务器的指标后，刷新整体的指标文件
    with open(metric_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_metrics, json_file, ensure_ascii=False, indent=4)

    # 将指标内容更新进知识库
    update_vector_db(all_metrics)


if __name__ == '__main__':
    start_main_time = time.perf_counter()
    main()
    end_main_time = time.perf_counter()
    print(f'Run time: {end_main_time - start_main_time:.4f}s')

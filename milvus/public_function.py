"""
公共函数库
该文件包含一些常用的公共函数，用于数据处理、文件读写等操作。
"""

import os
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from processor.SummaryGenerator import SummaryGenerator


def write_data_to_pkl(docs, file_path):
    """将数据写入到 pkl 文件"""
    with open(file_path, 'wb') as file:
        pickle.dump(docs, file)


def read_data_from_pkl(file_path):
    """从 pkl 文件加载数据 """
    try:
        with open(file_path, 'rb') as file:
            docs = pickle.load(file)
    except:
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    return docs


def write_data_to_txt(data, file_path):
    """将数据写入到 txt 文件"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def read_data_from_txt(file_path):
    """从 txt 文件加载数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except:
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    return data


def get_local_model_path(model_name: str) -> str:
    """
    将 'Alibaba-NLP/gte-multilingual-base' 转换成本地 cache 路径
    如 ~/.cache/huggingface/hub/models--Alibaba-NLP--gte-multilingual-base/snapshots/<hash>
    """
    safe_name = model_name.replace("/", "--")
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hub_dir / f"models--{safe_name}" / "snapshots"
    
    # 默认只取第一个 snapshot 子目录（一般只有一个）
    snapshot_dirs = list(model_dir.glob("*"))
    if not snapshot_dirs:
        raise FileNotFoundError(f"No local snapshot found for model: {model_name}")
    
    return str(snapshot_dirs[0])


def load_embedding_model(model_name: str, device: str):
    """加载嵌入模型"""
    print(f"🚀 嵌入模型：{model_name}，努力加载中...")
    try:
        model_path = get_local_model_path(model_name)
        embedding_model = SentenceTransformer(
            model_path,
            device = device,
            trust_remote_code = True,
            local_files_only = True
        )
        print("✅ 嵌入模型加载完成！")
    except Exception as e:
        print(f"❌ 嵌入模型加载失败：{e}")
        raise e

    return embedding_model


def load_summary_model(model_name: str, config):
    """加载摘要生成模型"""
    print(f"🚀 摘要生成模型：{model_name}，努力加载中...")
    try:
        summary_model = SummaryGenerator(model_name, config)
        print("✅ 摘要生成模型加载完成！")
    except Exception as e:
        print(f"❌ 摘要生成模型加载失败：{e}")
        raise e

    return summary_model



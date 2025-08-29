import json
import numpy as np
from openai import OpenAI
from typing import Dict, List, Tuple


client = OpenAI(
    base_url="http://a6000-G5500-V6:5415/v1",
    api_key="EMPTY",
)


class SimilarityMetric:
    # 定义支持的相似度计算方法
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    PEARSON = "pearson"

    @staticmethod
    def calculate(v1: np.ndarray, v2: np.ndarray, metric: str = "cosine") -> float:
        if metric == SimilarityMetric.COSINE:
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif metric == SimilarityMetric.EUCLIDEAN:
            distance = np.sqrt(np.sum((v1 - v2) ** 2))
            return 1 / (1 + distance)
        elif metric == SimilarityMetric.MANHATTAN:
            distance = np.sum(np.abs(v1 - v2))
            return 1 / (1 + distance)
        elif metric == SimilarityMetric.JACCARD:
            threshold = 0.5
            set1 = set(np.where(v1 > threshold)[0])
            set2 = set(np.where(v2 > threshold)[0])
            if not set1 and not set2:
                return 0.0
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union
        elif metric == SimilarityMetric.PEARSON:
            return np.corrcoef(v1, v2)[0, 1]
        else:
            raise ValueError(f"不支持的相似度计算方法: {metric}")


class Tool:
    def __init__(self, name: str, func, description: str):
        self.name = name          # 工具名称
        self.func = func          # 工具函数
        self.description = description  # 工具描述
        self._embedding = None    # 描述文本的向量表示


    @property
    def embedding(self):
        # 懒加载方式获取向量表示
        if self._embedding is None:
            self._embedding = query_embedding(self.description)
        return self._embedding


class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}  # 工具注册表
        self.similarity_threshold = 0.7  # 相似度阈值
        self.similarity_metric = SimilarityMetric.COSINE  # 默认使用余弦相似度


    def register_tool(self, name: str, func, description: str):
        """
        注册工具
        :param name: 工具名称
        :param func: 工具对应的函数
        :param description: 工具功能描述
        """
        self.tools[name] = Tool(name, func, description)


    def find_matching_tools(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        找到与查询最匹配的工具
        :param query: 用户查询
        :param top_k: 返回前k个最匹配的工具
        :return: [(tool_name, similarity_score), ...]
        """
        query_emb = query_embedding(query)
        
        similarities = []
        for name, tool in self.tools.items():
            similarity = SimilarityMetric.calculate(
                query_emb, 
                tool.embedding,
                self.similarity_metric
            )
            similarities.append((name, similarity))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 过滤掉相似度低于阈值的结果
        filtered = [(name, score) for name, score in similarities[:top_k] 
                    if score >= self.similarity_threshold]
        
        return filtered


    def call_tool(self, name: str, *args, **kwargs):
        # 调用指定工具
        if name not in self.tools:
            raise ValueError(f"工具 {name} 未注册")
        return self.tools[name].func(*args, **kwargs)


def query_embedding(text: str) -> np.ndarray:
    """获取文本的嵌入向量"""
    response = client.embeddings.create(
        model="Alibaba-NLP/gte-multilingual-base",
        input=[text],
        encoding_format="float"
    )
    embedding = json.loads(response.model_dump_json())['data'][0]['embedding']
    return np.array(embedding)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# 初始化工具管理器
tool_manager = ToolManager()


# 注册工具（使用更丰富的描述）
tool_manager.register_tool(
    name = "disk_usage",
    func = lambda: ["df -h"],
    description = "查看文件系统存储状况：各分区的容量、使用量及挂载点信息"
)


tool_manager.register_tool(
    name = "memory_usage",
    func = lambda: ["free -m"],
    description = "查看系统内存状况：物理内存和交换分区的使用情况"
)


tool_manager.register_tool(
    name = "gpu_usage",
    func = lambda: ["nvidia-smi"],
    description = "查看 NVIDIA 显卡状况：GPU 负载、显存占用、温度和运行进程"
)


tool_manager.register_tool(
    name = "slurm_queue",
    func = lambda: ["yhq"],
    description = ["作业队列", "任务排队", "有哪些任务在排队", "当前有哪些作业"]
)


# tool_manager.register_tool(
#     name = "my_slurm_jobs",
#     func = lambda: ["yhq -u $USER"],
#     description = ["我的作业", "我提交的作业", "我排的队", "我的任务"]
# )


# tool_manager.register_tool(
#     name = "slurm_nodes_status",
#     func = lambda: ["yhi"],
#     description = ["节点状态", "节点信息", "集群节点", "有多少节点", "哪些节点在运行"]
# )


# tool_manager.register_tool(
#     name = "slurm_nodes_status",
#     func = lambda: ["yhacct"],
#     description = ["节点状态", "节点信息", "集群节点", "有多少节点", "哪些节点在运行"]
# )


# tool_manager.register_tool(
#     name = "slurm_nodes_status",
#     func = lambda: ["yhalloc"],
#     description = ["节点状态", "节点信息", "集群节点", "有多少节点", "哪些节点在运行"]
# )


# tool_manager.register_tool(
#     name = "slurm_nodes_status",
#     func = lambda: ["yhattach"],
#     description = ["节点状态", "节点信息", "集群节点", "有多少节点", "哪些节点在运行"]
# )


# tool_manager.register_tool(
#     name = "slurm_nodes_status",
#     func = lambda: ["yhbatch"],
#     description = ["节点状态", "节点信息", "集群节点", "有多少节点", "哪些节点在运行"]
# )


# tool_manager.register_tool(
#     name = "slurm_nodes_status",
#     func = lambda: ["yhbcast"],
#     description = ["节点状态", "节点信息", "集群节点", "有多少节点", "哪些节点在运行"]
# )


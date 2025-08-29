import json
from deepeval.models import OllamaModel
import pandas as pd

# 读取 JSON 文件
with open('./evaluation_metagpt_results/resultsumo1-210.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化一个空列表，用于存储每个用例的指标得分
rows = []

# 遍历每个测试用例
for test_case in data['test_results']:
    # 初始化一个字典，用于存储当前用例的指标得分
    scores = {}
    # 遍历当前用例的每个指标
    for metric in test_case['metrics_data']:
        # 获取指标名称和得分
        metric_name = metric['name']
        score = metric['score']
        # 将指标得分添加到字典中
        scores[metric_name] = score
    # 将当前用例的指标得分添加到列表中
    rows.append(scores)

# 将列表转换为 DataFrame
df = pd.DataFrame(rows)

# 将 DataFrame 写入 XLS 文件
df.to_excel('./test_result.xlsx', index=False)

print("数据已成功写入 test_result.xlsx 文件。")
from deepeval import evaluate,assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric,ContextualPrecisionMetric,ContextualRelevancyMetric,ContextualRecallMetric    
from deepeval.dataset import EvaluationDataset
import json
import pandas as pd

file = 'conversation_data_naiverag.json' #每次更新数据文件名
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化空的 EvaluationDataset
dataset = EvaluationDataset()

# 遍历数据并逐个添加到 dataset
for item in data:
    test_case = LLMTestCase(
        input=item["input"],
        actual_output=item["actual_output"],
        expected_output=item["expected_output"],
        retrieval_context=item["retrieval_context"]
    )
    dataset.add_test_case(test_case) 

#def test_metrics():
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5,include_reason=True,verbose_mode=True)
faithfulness_metric = FaithfulnessMetric(threshold=0.5,include_reason=True,verbose_mode=True)
precision_metric = ContextualPrecisionMetric(threshold=0.5, include_reason=True,verbose_mode=True)
relevancy_metric = ContextualRelevancyMetric(threshold=0.5, include_reason=True,verbose_mode=True)
recall_metric = ContextualRecallMetric(threshold=0.5, include_reason=True,verbose_mode=True)

result = evaluate(dataset, [answer_relevancy_metric, faithfulness_metric, precision_metric, relevancy_metric])
# 将结果转换为字典（使用递归方式处理所有嵌套对象）
def to_dict(obj):
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return obj

result_dict = to_dict(result)

# 保存为单个JSON文件，再使用test_metric.py文件转换为表格
with open('./evaluation_light_results/result.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)

print("评估结果已保存至: ./evaluation_light_results/result.json")
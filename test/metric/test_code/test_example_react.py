from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall, ConversationalTestCase
from deepeval.metrics import TaskCompletionMetric,ToolCorrectnessMetric,AnswerRelevancyMetric,HallucinationMetric,ConversationRelevancyMetric,RoleAdherenceMetric
import json

file = '1.json'
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = []
for item in data:
    test_case = LLMTestCase(
        input=item["input"],
        actual_output=item["actual_output"],
        expected_output=item["expected_output"],
        tools_called=[
            ToolCall(
                name=item["toolcall"][0]["name"],
                description=item["toolcall"][0]["description"],
                input_parameters=item["toolcall"][0]["input_parameters"],
                output=item["toolcall"][0]["output"]
            )
        ],
        context=item["toolcall"][0]["output"],
    )
    turns_react = [test_case]
    convo_test_case = ConversationalTestCase(
        chatbot_role="要求必须调用tool，首先依据tool返回的内容，总结和问题相关的信息，写一段总结性文字。然后依据大模型的自身的运维经验和指令，总结和问题相关的信息。结合两方面内容，回答问题。当前状态不确定的，给出查询的具体解决方案和指令。",
        turns=turns_react
    )
    dataset.append(convo_test_case) 

roleAdherence_metric = RoleAdherenceMetric(threshold=0.5,include_reason = True,verbose_mode = True)
conversationrelevancy_metric = ConversationRelevancyMetric(threshold=0.5,include_reason = True,verbose_mode = True)
completion_metric = TaskCompletionMetric(threshold=0.5, include_reason=True, verbose_mode=True)
hallucination_metric = HallucinationMetric(threshold=0.5, include_reason=True, verbose_mode=True)

result = evaluate(dataset, [roleAdherence_metric, conversationrelevancy_metric, completion_metric, hallucination_metric])

def to_dict(obj):
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return obj

result_dict = to_dict(result)

# 保存为单个JSON文件，再使用test_metric.py文件转换为表格
with open('./evaluation_react_results/result.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)

print("评估结果已保存至: ./evaluation_react_results/result.json")

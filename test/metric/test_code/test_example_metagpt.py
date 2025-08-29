from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric, ConversationalGEval, ConversationCompletenessMetric, ConversationRelevancyMetric
import json

file = 'conversation_data_metagpt_sumo.json' #每次更新数据文件名
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = []
for item in data:
    test_case = LLMTestCase(
    input=item["input"],
    actual_output=item["actual_output"]
    )
    turns_meta = [test_case]
    convo_test_case = ConversationalTestCase(
        chatbot_role="根据给出的信息回答问题，要求回答是一段长文字，可以回答技术细节或常识性内容，也可以回答操作指令或解决方案，还可以给出建议。所有可以的选项不强制要求回答。",
        turns=turns_meta
    )
    dataset.append(convo_test_case) 

roleAdherence_metric = RoleAdherenceMetric(threshold=0.5, include_reason = True, verbose_mode = True)
relevancy_metric = ConversationRelevancyMetric(threshold=0.5,include_reason = True,verbose_mode = True)
completion_metric = ConversationCompletenessMetric(threshold=0.5,include_reason = True,verbose_mode = True)

professionalism_metric = ConversationalGEval(
    name="Professionalism",
    criteria="""给定'actual_output'是LLM生成的响应，'input'是用户提出的查询，判断LLM在整个对话过程中是否表现得专业。""",
    evaluation_steps=[
        "检查每个LLM生成的'actual_output'相对于用户的查询'input'是否专业",
        "专业意味着无脏话、无幻觉语言，语气严肃认真，不带任何负面情绪。",
        "用中文回复"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    verbose_mode=True
)

result = evaluate(dataset, [roleAdherence_metric, relevancy_metric, completion_metric, professionalism_metric])

def to_dict(obj):
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if hasattr(obj, '__dict__'):
        return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return obj

result_dict = to_dict(result)

# 保存为单个JSON文件，再使用test_metric.py文件转换为表格
with open('./evaluation_metagpt_results/resultsumo.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)

print("评估结果已保存至: ./evaluation_metagpt_results/resultsumo.json")

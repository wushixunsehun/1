import sys
sys.path.append('./')
import csv
from tqdm import tqdm
from agentsAPI import query_llm, strip_think
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


PROMPT_TEMPLATE_v1 = '''
你是一位专业的智能系统任务决策师，任务是根据用户提供的任务描述，准确判断任务类型并输出结果。请严格按照以下要求完成任务分类：

任务类型及关键特征：
- 0：知识问答任务——接收问题，检索知识库（RAG），组织答案，输出结果。
- 1：状态查询任务——需要获取服务器状态，生成命令，执行终端操作，整理返回数据，生成报告。
- 2：异常处理任务——出现异常，获取对象服务器状态和数据库数据，分析异常，制定解决方案，执行解决方案，生成报告。

要求：
- 仅根据用户提供的任务描述进行判断，不引入额外主观假设。
- 只返回单独的数字（0、1或2），无需包含思考过程等其余多余内容。

请根据任务描述：{task}
严格按照上述流程推理，并最终只输出数字。
'''

PROMPT_TEMPLATE_v1_COT = '''
你是一位专业的智能系统任务决策师，任务是根据用户提供的任务描述，准确判断任务类型并输出结果。请严格按照以下要求完成任务分类：

任务类型及关键特征：
- 0：知识问答任务——接收问题，检索知识库（RAG），组织答案，输出结果。
- 1：状态查询任务——需要获取服务器状态，生成命令，执行终端操作，整理返回数据，生成报告。
- 2：异常处理任务——出现异常，获取对象服务器状态和数据库数据，分析异常，制定解决方案，执行解决方案，生成报告。

推理流程（请依次思考）：
1. 是否是知识性问题，如“如何”、“什么是”等关键词？
2. 是否在请求实时状态，如“检查状态”、“运行时间”等？
3. 是否明确指出异常、报错、故障，并寻求修复方案？
4. 若以上皆不满足，回归任务关键词进行字面分析，匹配最贴近的任务类型定义。
5. 禁止引入主观臆测或虚构场景。

输出要求：
- 只返回单独的数字（0、1或2）
- 不包含任何解释、标签或其他内容

请根据任务描述：{task}
严格按照上述推理流程判断，并最终只输出数字。
'''

PROMPT_TEMPLATE_v2 = '''
你是一位专业的智能系统任务决策师，你需要根据用户提供的任务描述，分析用户输入的意图，准确判断任务类型。请严格按照以下要求完成任务分类：

任务类型及关键特征：
- 0：知识问答任务——用户提出问题，系统需检索知识库（RAG），整理并输出答案。通常涉及知识查询、解释、背景介绍等，不涉及实际操作或状态获取。
- 1：状态查询任务——用户需要了解服务器、服务或系统的当前状态。系统需生成命令，执行终端操作，收集并整理返回数据，最后生成状态报告。
- 2：异常处理任务——用户描述出现异常或故障，需要解决。系统需获取相关服务器状态和数据库数据，分析异常原因，制定并执行解决方案，最后生成处理报告。

要求：
- 仅根据用户提供的任务描述进行判断，不引入额外主观假设。
- 只返回单独的数字（0、1或2），无需包含思考过程等其余多余内容。

请根据任务描述：{task}
严格按照上述流程推理，并最终只输出数字。
'''

PROMPT_TEMPLATE_v2_COT = '''
你是一位专业的智能系统任务决策师，你需要根据用户提供的任务描述，分析用户输入的意图，准确判断任务类型。请严格按照以下要求完成任务分类：

任务类型及关键特征：
- 0：知识问答任务——用户提出问题，系统需检索知识库（RAG），整理并输出答案。通常涉及知识查询、解释、背景介绍等，不涉及实际操作或状态获取。
- 1：状态查询任务——用户需要了解服务器、服务或系统的当前状态。系统需生成命令，执行终端操作，收集并整理返回数据，最后生成状态报告。
- 2：异常处理任务——用户描述出现异常或故障，需要解决。系统需获取相关服务器状态和数据库数据，分析异常原因，制定并执行解决方案，最后生成处理报告。

推理流程（请逐步判断）：
1. 识别用户是否在“提问知识”或“解释概念”（如使用“如何”、“什么是”、“为什么”等）
2. 检查是否为“查询状态”，如含有“查看”、“获取”、“运行状态”、“时间”等实时信息相关词汇
3. 判断是否存在“异常”或“报错”或“故障”之类的描述，并明确需要“解决”
4. 若不符合任何单项定义，请优先匹配字面意义最接近的一项，避免主观推断

输出要求：
- 只返回单独的数字（0、1或2）
- 不包含任何其他内容

请根据任务描述：{task}
严格按照上述推理流程判断，并最终只输出数字。
'''

PROMPT_TEMPLATE_v3 = '''
你是一位专业的智能系统任务决策专家，请严格根据用户输入的任务描述进行意图分析，并精确匹配到以下任务类型：

### 任务类型定义
0. **知识问答任务**
    - 特征：用户询问事实性/解释性问题。
    - 系统行为：检索知识库(RAG)，整理信息并输出答案
    - 示例：
        "如何安装 xx 软件/硬件？"
        "如何查看 xx 节点的状态？"

1. **状态查询任务**
    - 特征：用户请求获取系统/服务的实时状态信息
    - 系统行为：生成终端命令→执行操作→整理状态报告
    - 示例：
        "检查 xx 服务的运行状态"
        "xx 作业的活跃时长是多少？"

2. **异常处理任务**
    - 特征：用户报告故障/异常，需要诊断和解决
    - 系统行为：获取系统状态→分析原因→执行修复→生成处理报告
    - 示例：
        "磁盘出现报错，分析原因给出解决方案"
        "xx 任务运行中断，请检查原因"

### 分析规则
1. 严格基于文本字面含义分析，禁止主观推测
2. 无状态操作的知识请求 ≠ 状态查询
3. 单纯询问解决方案 ≠ 异常处理（需实际故障描述）

### 输出要求
- 仅返回单个数字（0/1/2）
- 禁止任何解释或附加文本

请分析任务：{task}
'''

PROMPT_TEMPLATE_v3_COT = '''
你是一位专业的智能系统任务决策专家，请严格根据用户输入的任务描述进行意图分析，并使用**思维链推理**的方法，逐步判断其最符合哪一类任务类型。请严格遵循以下任务定义和分析规则，最终仅返回对应的任务类型编号（0/1/2）。

### 任务类型定义
0. **知识问答任务**
    - 特征：用户询问事实性或解释性问题，目的是获取知识或说明
    - 系统行为：检索知识库(RAG)，整理信息并输出答案
    - 示例：
        "如何安装 xx 软件/硬件？"
        "如何查看 xx 节点的状态？"

1. **状态查询任务**
    - 特征：用户请求获取某个系统或服务的实时状态或执行状态
    - 系统行为：生成命令 → 获取实时状态 → 输出状态报告
    - 示例：
        "检查 xx 服务的运行状态"
        "xx 作业的活跃时长是多少？"

2. **异常处理任务**
    - 特征：用户报告某种异常或故障，目的是找出原因并提出修复方案
    - 系统行为：收集状态 → 分析根因 → 执行修复 → 输出处理流程与结论
    - 示例：
        "磁盘出现报错，分析原因给出解决方案"
        "xx 任务运行中断，请检查原因"

### 分析规则（务必逐步思考）
1. **从字面上识别用户的意图关键词和语气（如‘如何’、‘查看’、‘中断’、‘报错’等）**
2. **判断是否涉及对系统或服务“当前状态”的获取或报告**
3. **判断是否明确描述了“异常”或“故障”的现象，并请求解决方案**
4. **排除项：
    - 知识请求 ≠ 状态查询（无运行环境或系统上下文）
    - 请求建议或方法 ≠ 异常处理（缺少故障描述）**

### 输出要求
- 请使用思维链（Chain-of-Thought）进行推理，并在思考完成后，**仅返回最终的数字编号（0/1/2）**
- **不得输出任何额外文本、解释或标签**

请分析任务：{task}
'''


PROMPT_TEMPLATE_v4 = '''你是一位专业的智能系统任务决策专家，请严格根据用户输入的任务描述进行意图分析，并精确匹配到以下任务类型：

### 任务类型定义
0. **知识问答任务**
    - 特征：用户询问事实性/解释性问题。
    - 系统行为：检索知识库(RAG)，整理信息并输出答案
    - 示例：
        "如何安装 xx 软件/硬件？"
        "如何查看 xx 节点的状态？"

1.  **状态查询任务**
    - 特征：用户请求获取系统/服务的实时状态信息
    - 系统行为：生成终端命令→执行操作→整理状态报告
    - 示例：
        "检查 xx 服务的运行状态"
        "xx 作业的活跃时长是多少？"

2. **异常处理任务**
    - 特征：用户报告故障/异常，需要诊断和解决
    - 系统行为：获取系统状态→分析原因→执行修复→生成处理报告
    - 示例：
        "磁盘出现报错，分析原因给出解决方案"
        "xx 任务运行中断，请检查原因"

3. **写操作任务**
    - 特征：用户需要对系统资源进行“写”操作，如修改/创建文件、更新配置等
    - 系统行为：基于任务描述直接调用符合需求的脚本/运维剧本→执行→返回结果确认
    - 示例：
        "把 xx 中的 xx 调成 xx"
        "修改用户 xx 的作业配额"
        "取消作业 xx"

### 分析规则
1. 严格基于文本字面含义分析，禁止主观推测
2. 无状态操作的知识请求 ≠ 状态查询
3. 单纯询问解决方案 ≠ 异常处理（需实际故障描述）
4. 对系统资源的变更操作优先归类为写操作任务

### 输出要求
- 仅返回单个数字（0/1/2/3）
- 禁止任何解释或附加文本

请分析任务：{task}
'''


OUT_FILE = 'pre_result_v3.csv'
def call_llm_api(text):
    prompt = PROMPT_TEMPLATE_v3.format(task=text)
    enable_thinking = False
    if enable_thinking:
        response = query_llm(prompt, enable_thinking=enable_thinking)
        response = strip_think(response).strip()
    else:
        response = query_llm(prompt)

    return response


def read_questions_and_labels(filepath):
    questions = []
    y_true = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row['text'])
            y_true.append(int(row['true_label']))
    return questions, y_true


def get_predictions(questions):
    y_pred = []
    for text in tqdm(questions):
        pred = call_llm_api(text)
        # 只保留数字部分
        try:
            pred = int(str(pred).strip())
        except Exception:
            pred = 0
        y_pred.append(pred)
    return y_pred


def evaluate(y_true, y_pred):
    if y_true:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        confusion_matrix_result = confusion_matrix(y_true, y_pred)
        print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}')
        print('Confusion Matrix:')
        print(confusion_matrix_result)
    else:
        print('true_label为空或无有效标签')


def write_result_csv(questions, y_pred, y_true, input_csv_path, output_csv_path):
    with open(input_csv_path, 'r', encoding='utf-8') as fin, \
        open(output_csv_path, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        fieldnames = ['id', 'text', 'true_label', 'pred_label']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for _, (row, pred, true) in enumerate(zip(reader, y_pred, y_true)):
            writer.writerow({
                'id': row['id'],
                'text': row['text'],
                'true_label': true,
                'pred_label': pred
            })


def main():
    input_csv = 'task_acc_test/data/groundtruth.csv'
    output_csv = f'task_acc_test/result/{OUT_FILE}'
    questions, y_true = read_questions_and_labels(input_csv)
    y_pred = get_predictions(questions)
    evaluate(y_true, y_pred)
    write_result_csv(questions, y_pred, y_true, input_csv, output_csv)
    print(f'Result file saved to: {output_csv}')


if __name__ == '__main__':
    main()

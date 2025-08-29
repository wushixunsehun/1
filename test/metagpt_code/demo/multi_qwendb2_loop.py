import asyncio
import re
import fire
import os
import time
import json
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
import tiktoken
from typing import Dict, List, Optional

# 初始化cl100k_base编码
encoding = tiktoken.get_encoding("cl100k_base")

# Token统计器
class TokenTracker:
    def __init__(self):
        self.role_token_stats: Dict[str, List[Dict[str, int]]] = {}  # 按角色存储每轮token数据

    def count_tokens(self, text: str, role: str, round_num: int, is_input: bool) -> int:
        token_count = len(encoding.encode(text))
        if role not in self.role_token_stats:
            self.role_token_stats[role] = []
        self.role_token_stats[role].append({
            "round": round_num,
            "type": "input" if is_input else "output",
            "tokens": token_count
        })
        return token_count

token_tracker = TokenTracker()

# 时间统计器
class TimeTracker:
    def __init__(self):
        self.role_time_stats: Dict[str, List[Dict[str, float]]] = {}  # 按角色存储每轮时间数据

    def start_timer(self, role: str, round_num: int):
        """记录开始时间"""
        if role not in self.role_time_stats:
            self.role_time_stats[role] = []
        self.role_time_stats[role].append({
            "round": round_num,
            "start_time": time.perf_counter()
        })

    def stop_timer(self, role: str, round_num: int):
        """计算并记录结束时间"""
        if role in self.role_time_stats:
            for entry in self.role_time_stats[role]:
                if entry["round"] == round_num and "end_time" not in entry:
                    entry["end_time"] = time.perf_counter()

    def get_round_time(self, role: str, round_num: int) -> float:
        """获取某轮的时间差（秒）"""
        if role in self.role_time_stats:
            for entry in self.role_time_stats[role]:
                if entry["round"] == round_num:
                    return entry.get("end_time", 0) - entry.get("start_time", 0)
        return 0.0

time_tracker = TimeTracker()

def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    return match.group(1) if match else rsp

class Simpleqwen(Action):
    PROMPT_TEMPLATE: str = """
    问题: {user_question}
    
    你正在与{opponent_name}参加一个研讨会。
    你是第一个发表意见的人。
    请依据大模型的自身的运维经验和指令，用中文简要陈述你的观点，不限制字数。
    当前状态不确定的，给出查询的具体解决方案和指令。
    要求回答是一段长文字。
    鼓励列点回答(标注1.2.3.)，文字不能加粗，列点前要求有综述性的文字和冒号，不要换行或者分段。
    返回``` opinion of qweno ```。
    你的观点:
    """

    name: str = "Simpleqwen"

    async def run(self, opponent_name: str, round_num: int, user_question: str, output_file: str):
        # 记录开始时间
        time_tracker.start_timer("qweno", round_num)
        
        prompt = self.PROMPT_TEMPLATE.format(opponent_name=opponent_name, user_question=user_question)
        input_tokens = token_tracker.count_tokens(prompt, "qweno", round_num, is_input=True)
        
        rsp = await self._aask(prompt)
        output_tokens = token_tracker.count_tokens(rsp, "qweno", round_num, is_input=False)
        
        # 记录结束时间
        time_tracker.stop_timer("qweno", round_num)
        
        viewpoint_text = parse_code(rsp)
        round_time = time_tracker.get_round_time("qweno", round_num)
        logger.info(f"qweno Round {round_num}: Tokens={input_tokens+output_tokens}, 耗时={round_time:.2f}秒")
        return viewpoint_text

class Simpleqwener(Role):
    name: str = "qweno"
    profile: str = "Simpleqwener"

    def __init__(self, user_question: str, output_file: str, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([Simpleqwen()])
        self._watch([UserRequirement])
        self.round_num = 1
        self.user_question = user_question
        self.output_file = output_file

    async def _act(self) -> Message:
        rsp = await self.rc.todo.run(
            opponent_name="dbo", 
            round_num=self.round_num,
            user_question=self.user_question,
            output_file=self.output_file
        )

        # 修改为追加模式，并添加角色标识和分隔符
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("===== QWENO观点 =====\n")  # 明确的角色标识
            rsp_o = rsp.replace('\n', '')
            f.write(rsp_o)
            f.write('\n\n')  # 空行分隔
            f.flush()

        self.round_num += 1
        return Message(content=rsp, role=self.profile, cause_by=type(self.rc.todo))

class Simpledb(Action):
    PROMPT_TEMPLATE: str = """
    问题: {user_question}
    
    你正在与{opponent_name}参加一个研讨会。
    你是第二个发表意见的人。
    在qweno发表意见后，你应该基于获取的txt文件陈述你的观点。用中文简要陈述你的观点，不限制字数。
    要求回答是一段长文字。
    鼓励列点回答(标注1.2.3.)，文字不能加粗，列点前要求有综述性的文字和冒号，不要换行或者分段。
    请用中文简要陈述你的观点，不限制字数。
    返回``` opinions of the dbo ```。
    你的观点:
    """

    name: str = "Simpledb"
    
    def read_response_file(self, input_file: str):
        if os.path.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as f:
                return f.read()
        return f"未找到{input_file}文件"

    async def run(self, opponent_name: str, round_num: int, qweno_opinion: str, user_question: str, 
                  input_file: str, output_file: str):
        time_tracker.start_timer("dbo", round_num)
        
        file_content = self.read_response_file(input_file)
        prompt = (
            f"qweno的观点: {qweno_opinion}\n"
            f"{input_file}文件内容: {file_content}\n\n"
            f"{self.PROMPT_TEMPLATE.format(opponent_name=opponent_name, user_question=user_question)}"
        )
        
        input_tokens = token_tracker.count_tokens(prompt, "dbo", round_num, is_input=True)
        rsp = await self._aask(prompt)
        output_tokens = token_tracker.count_tokens(rsp, "dbo", round_num, is_input=False)
        
        time_tracker.stop_timer("dbo", round_num)
        
        db_text = parse_code(rsp)
        round_time = time_tracker.get_round_time("dbo", round_num)
        logger.info(f"dbo Round {round_num}: Tokens={input_tokens+output_tokens}, 耗时={round_time:.2f}秒")
        return db_text

class Simpledber(Role):
    name: str = "dbo"
    profile: str = "Simpledber"

    def __init__(self, user_question: str, input_file: str, output_file: str, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([Simpledb()])
        self._watch([UserRequirement])
        self.round_num = 1
        self.qweno_opinion = ""
        self.user_question = user_question
        self.input_file = input_file
        self.output_file = output_file

    async def _act(self) -> Message:
        for mem in self.get_memories():
            if mem.role == "Simpleqwener":
                self.qweno_opinion = mem.content
        rsp = await self.rc.todo.run(
            opponent_name="qweno",
            round_num=self.round_num,
            qweno_opinion=self.qweno_opinion,
            user_question=self.user_question,
            input_file=self.input_file,
            output_file=self.output_file
        )

        # 保持追加模式，并添加角色标识和分隔符
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("===== DBO观点 =====\n")  # 明确的角色标识
            rsp_o = rsp.replace('\n', '')
            f.write(rsp_o)
            f.write('\n\n')  # 空行分隔
            f.flush()

        self.round_num += 1
        return Message(content=rsp, role=self.profile, cause_by=type(self.rc.todo))

class Simplesummarize(Action):
    PROMPT_TEMPLATE: str = """
    问题: {user_question}
    
    Context: {context}
    在qweno和dbo发表所有意见后，你应该基于所有观点总结研讨会结果。
    结合两者的信息，用中文简要陈述你的结论，不限制字数。
    要求回答是一段长文字。
    鼓励列点回答(标注1.2.3.)，文字不能加粗，列点前要求有综述性的文字和冒号，不要换行或者分段。
    返回``` opinion of sumo ```。
    你的陈述:
    """

    name: str = "Simplesummarize"

    async def run(self, context: str, round_num: int, user_question: str, output_file: str):
        time_tracker.start_timer("sumo", round_num)
        
        prompt = self.PROMPT_TEMPLATE.format(context=context, user_question=user_question)
        input_tokens = token_tracker.count_tokens(prompt, "sumo", round_num, is_input=True)
        rsp = await self._aask(prompt)
        output_tokens = token_tracker.count_tokens(rsp, "sumo", round_num, is_input=False)
        
        time_tracker.stop_timer("sumo", round_num)
        
        statement_text = parse_code(rsp)
        round_time = time_tracker.get_round_time("sumo", round_num)
        logger.info(f"sumo Round {round_num}: Tokens={input_tokens+output_tokens}, 耗时={round_time:.2f}秒")
        return statement_text

class Simplesummarizer(Role):
    name: str = "sumo"
    profile: str = "Simplesummarizer"

    def __init__(self, user_question: str, input_files: List[str], output_file: str, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([Simplesummarize()])
        self._watch([Simpleqwen, Simpledb])
        self.round_num = 1
        self.user_question = user_question
        self.input_files = input_files
        self.output_file = output_file

    async def _act(self) -> Message:
        qweno_msg = None
        dbo_msg = None
        
        # 尝试多次获取消息
        for attempt in range(3):  
            memories = self.get_memories()
            for mem in memories:
                if mem.role == "Simpleqwener":
                    qweno_msg = mem.content
                elif mem.role == "Simpledber":
                    dbo_msg = mem.content
                    
            if qweno_msg and dbo_msg:
                break
                
            # 等待一段时间再重试
            await asyncio.sleep(0.5)
            
        if not qweno_msg or not dbo_msg:
            logger.warning(f"sumo角色未能获取到所有需要的消息：qweno={bool(qweno_msg)}, dbo={bool(dbo_msg)}")
            # 处理缺失消息的情况
            if qweno_msg:
                context = qweno_msg
            elif dbo_msg:
                context = dbo_msg
            else:
                context = "没有足够的信息进行总结"
        else:
            context = f"{qweno_msg}\n{dbo_msg}"
            
        todo_task = self.rc.todo
        code_text = await todo_task.run(context=context, round_num=self.round_num, 
                                       user_question=self.user_question, output_file=self.output_file)
        
        # 保持追加模式，并添加角色标识和分隔符
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("===== SUMO总结 =====\n")  # 明确的角色标识
            code_text_o = code_text.replace('\n', '')
            f.write(code_text_o)
            f.write('\n\n')  # 空行分隔
            f.flush()

        self.round_num += 1
        return Message(content=code_text, role=self.profile, cause_by=type(todo_task))

async def process_file(file_idx: int, n_dialog_round: int = 2, investment: float = 3.0):
    """
    处理单个文件的函数，file_idx控制输入文件编号
    n_dialog_round保留为原有对话循环参数
    """
    input_file = f"./file_response4/response{file_idx}.txt"
    output_file = f"./file_answer4/answer{file_idx}.txt"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 初始化输出文件（清空并添加头部）
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"===== 文件 {file_idx} 处理结果 =====\n\n")
    
    # 读取输入文件内容
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            idea = input_data.get("query", "未找到查询内容")
    except json.JSONDecodeError:
        logger.warning(f"文件 {input_file} 不是有效的JSON，尝试读取为文本")
        with open(input_file, "r", encoding="utf-8") as f:
            idea = f.read().strip()
    
    start_time = time.perf_counter()
    logger.info(f"Start processing file: {input_file}, Idea: {idea}")

    # 初始化团队并执行对话
    team = Team()
    team.hire([
        Simpleqwener(user_question=idea, output_file=output_file),
        Simpledber(user_question=idea, input_file=input_file, output_file=output_file),
        Simplesummarizer(user_question=idea, input_files=[input_file], output_file=output_file)
    ])

    team.invest(investment=investment)
    team.run_project(idea)
    await team.run(n_round=n_dialog_round)  # 原有对话循环参数

    # 写入统计信息到输出文件
    total_time = time.perf_counter() - start_time
    with open(output_file, "a", encoding="utf-8") as f:
        # 时间统计
        f.write("\n===== 角色时间统计 =====\n")
        for role, stats in time_tracker.role_time_stats.items():
            total_round_time = sum(entry["end_time"] - entry["start_time"] for entry in stats)
            f.write(f"角色: {role} | 总耗时: {total_round_time:.2f}秒\n")
        
        # 总耗时
        f.write(f"\n===== 总耗时 =====\n")
        f.write(f"整体流程耗时: {total_time:.2f}秒\n")

        # Token统计
        f.write("\n===== Token使用统计 =====\n")
        for role, stats in token_tracker.role_token_stats.items():
            total_input = sum(s["tokens"] for s in stats if s["type"] == "input")
            total_output = sum(s["tokens"] for s in stats if s["type"] == "output")
            f.write(f"Role: {role} | 总输入Tokens: {total_input} | 总输出Tokens: {total_output}\n")

    # 重置统计器以便下一个文件使用
    token_tracker.role_token_stats = {}
    time_tracker.role_time_stats = {}
    time.sleep(2)  # 等待文件写入完成

async def main(
    start_pos: int = 199,       
    n_files: int = 1,         
    n_dialog_round: int = 2,        
    investment: float = 3.0         
):
    file_indices = list(range(start_pos, start_pos + n_files))
    logger.info(f"开始处理文件: response{file_indices}.txt")
    
    # 顺序处理多个文件
    for idx in file_indices:
        logger.info(f"开始处理文件 {idx}...")
        try:
            await process_file(idx, n_dialog_round, investment)
            logger.info(f"文件 {idx} 处理完成")
        except Exception as e:
            logger.error(f"处理文件 {idx} 时出错: {str(e)}")
            # 可选：出错后继续处理下一个文件
            # continue

if __name__ == "__main__":
    fire.Fire(main)
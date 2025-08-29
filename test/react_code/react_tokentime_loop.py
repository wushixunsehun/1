import json
import time
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from transformers import pipeline

# Token计数器类 - 使用cl100k_base编码
class TokenCounter:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def count_input_tokens(self, text: str) -> int:
        tokens = self.encoding.encode(text)
        self.total_input_tokens += len(tokens)
        return len(tokens)
    
    def count_output_tokens(self, text: str) -> int:
        tokens = self.encoding.encode(text)
        self.total_output_tokens += len(tokens)
        return len(tokens)
    
    def get_total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

def process_file(input_file_path: str, output_file_path: str):
    """
    处理单个文件
    
    Args:
        input_file_path: 输入文件路径
        output_file_path: 输出文件路径
    """
    # 初始化模型和token计数器
    token_counter = TokenCounter()
    model = ChatOpenAI(
        model="qwen3-30b-a3b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-dd57df4920454339abff52fb87a11671",
        extra_body={"enable_thinking": False}
    )

    # 定义工具
    @tool
    def summarize_response_file(query: str) -> str:
        """
        指定文件内容
        
        Args:
            query (str): 用户请求
            
        Returns:
            str: 文件内容
        """
        start_time = time.perf_counter()
        
        try:
            with open(input_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # # 加载预训练的摘要模型
            # summarizer = pipeline("summarization", model="t5-base")
            # max_length1 = int(len(content) * 0.5) 
            # summary = summarizer(content, max_length=max_length1, min_length=100, do_sample=False)[0]['summary_text']

            input_tokens = token_counter.count_input_tokens(query)
            output_tokens = token_counter.count_output_tokens(content)
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"工具调用 - 输入Tokens: {input_tokens}, 输出Tokens: {output_tokens}, 耗时: {elapsed_time:.2f}秒")
            
            return f"{input_file_path} 内容：\n{content}"
            
        except FileNotFoundError:
            error_msg = f"错误：找不到{input_file_path}文件"
            token_counter.count_output_tokens(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"读取文件时出错: {str(e)}"
            token_counter.count_output_tokens(error_msg)
            return error_msg

    # 创建LangGraph Tool
    toolsum = [summarize_response_file]

    def custom_print_stream(stream):
        full_response = ""
        start_time = time.perf_counter()
        
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
                full_response += str(message)
            else:
                message.pretty_print()
                full_response += message.content
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        output_tokens = token_counter.count_output_tokens(full_response)
        print(f"模型输出 - Tokens: {output_tokens}, 耗时: {elapsed_time:.2f}秒")
        
        return full_response, elapsed_time

    try:
        # 创建 React 代理
        graphx = create_react_agent(model, tools=toolsum)

        # 构建用户问题
        question = "要求必须调用tool，首先依据tool返回的内容，总结和问题相关的信息，写一段总结性文字。\n"
        question += "然后依据大模型的自身的运维经验和指令，总结和问题相关的信息。\n"
        question += "结合两方面内容，回答问题。当前状态不确定的，给出查询的具体解决方案和指令。\n"
        question += "要求回答是一段长文字。命令需要给出具体指令。\n"
        question += "鼓励列点回答(标注1.2.3.)，文字不能加粗，列点前要求有综述性的文字和冒号，不要换行或者分段。\n"
        question += "问题是："

        with open(input_file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            query = input_data.get("query", "未找到查询内容")
        
        question += str(query)

        # 计算用户问题的tokens
        input_tokens = token_counter.count_input_tokens(question)
        print(f"用户问题 - Tokens: {input_tokens}")
        
        # 手动构建输入格式
        inputs = {"messages": [{"role": "user", "content": question}]}
        
        # 记录整体开始时间
        overall_start_time = time.perf_counter()
        
        # 获取响应和耗时
        result, stream_time = custom_print_stream(graphx.stream(inputs, stream_mode="values"))
        
        try:
            # 处理响应内容
            marker = "}"
            index = result.find(marker)
            processed_string = result[index + len(marker):] if index != -1 else result
            
            # 记录整体时间
            overall_end_time = time.perf_counter()
            overall_elapsed_time = overall_end_time - overall_start_time

            # 写入输出文件
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(processed_string.strip())
                file.write('\n')
                file.write(f"总输入Tokens: {token_counter.total_input_tokens}\n")
                file.write(f"总输出Tokens: {token_counter.total_output_tokens}\n")
                file.write(f"总Tokens使用量: {token_counter.get_total_tokens()}\n")
                file.write(f"整体生成耗时: {overall_elapsed_time:.2f}秒\n")
                
            print(f"回答已写入: {output_file_path}")
            
        except Exception as e:
            print(f"处理响应时发生错误: {e}")

        print("\n=== Token统计结果 ===")
        print(f"总输入Tokens: {token_counter.total_input_tokens}")
        print(f"总输出Tokens: {token_counter.total_output_tokens}")
        print(f"总Tokens使用量: {token_counter.get_total_tokens()}")
        print("\n=== 时间统计结果 ===")
        print(f"整体生成耗时: {overall_elapsed_time:.2f}秒")

    except Exception as e:
        print(f"程序执行错误: {e}")

def main(loop_count: int = 3, start_index: int = 1):
    """
    主函数：循环处理多个文件
    
    Args:
        loop_count: 循环次数（i）
        start_index: 起始位置（k）
    """
    for i in range(1, loop_count + 1):
        file_index = start_index + i - 1
        print(f"\n===== 处理第 {i}/{loop_count} 个文件 (索引: {file_index}) =====")
        
        input_file_path = f"./file_response3/response{file_index}.txt"
        output_file_path = f"./file_answer_react3/answer{file_index}.txt"
        
        process_file(input_file_path, output_file_path)

if __name__ == "__main__":
    # 自定义循环次数和起始位置
    main(
        loop_count=1,                        
        start_index=166,
    )
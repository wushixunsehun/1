import json
import time
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

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

def main(input_file_path: str = "response.txt", output_file_path: str = "answer1.txt"):
    """
    主函数：处理用户查询并生成回答
    
    Args:
        input_file_path: 输入文件路径（默认: response.txt）
        output_file_path: 输出文件路径（默认: answer1.txt）
    """
    # 初始化模型和token计数器
    token_counter = TokenCounter()
    model = ChatOpenAI(
        model="qwen3-4b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-b3d6320e7d3a47cc85a76c1dc8fdfdd5",
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
        question = "通过调用tool，依据tool返回的内容，回答问题。\n"
        question += "要求回答是一段长文字。如果有命令需要给出具体指令。\n"
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
                file.write(f"总输入Tokens: {token_counter.total_input_tokens}")
                file.write(f"总输出Tokens: {token_counter.total_output_tokens}")
                file.write(f"总Tokens使用量: {token_counter.get_total_tokens()}")
                file.write(f"流式输出耗时: {stream_time:.2f}秒")
                file.write(f"整体生成耗时: {overall_elapsed_time:.2f}秒")
                
            print(f"回答已写入: {output_file_path}")
            
        except Exception as e:
            print(f"处理响应时发生错误: {e}")

        print("\n=== Token统计结果 ===")
        print(f"总输入Tokens: {token_counter.total_input_tokens}")
        print(f"总输出Tokens: {token_counter.total_output_tokens}")
        print(f"总Tokens使用量: {token_counter.get_total_tokens()}")
        print("\n=== 时间统计结果 ===")
        print(f"流式输出耗时: {stream_time:.2f}秒")
        print(f"整体生成耗时: {overall_elapsed_time:.2f}秒")

    except Exception as e:
        print(f"程序执行错误: {e}")

if __name__ == "__main__":
    # 可自定义输入输出路径
    main(
        input_file_path="./file_response_react/response.txt",  # 输入文件路径
        output_file_path="./file_answer_react/answer.txt"    # 输出文件路径
    )
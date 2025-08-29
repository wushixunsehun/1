"""
任务解析agent(系统决策智能体)
系统感知agent(专家智能体:数据分析工程师) (时间序列数据库专家 日志数据库专家 系统状态专家)
异常分析Agent(专家智能体)(日志异常检测 时间序列异常检测)
策略规划Agent(专家智能体)+RAG（先匹配规则，没有规则的再由LLM规划）
操作执行Agent(专家智能体：运维工程师)：LLM模拟操作结果
报告反馈Agent(专家智能体)


"""
# coding=gb2312
import json
from typing import TypedDict, Annotated
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx

"""
任务解析agent(系统决策智能体)
1.按照需求生成DAG
2.根据DAG生成langgraph(执行链)执行
MCP协议格式:JSON-RPC 2.0规范
请求
{
  "jsonrpc": "2.0",
  "id": "string | number",
  "method": "string",
  "params": {
    "[key: string]": "unknown"
  }
}
响应
{
  "jsonrpc": "2.0",
  "id": "string | number",
  "result": {
    "[key: string]": "unknown"
  },
  "error": {
    "code": "number",
    "message": "string",
    "data": "unknown"
  }
}
通知
{
  "jsonrpc": "2.0",
  "method": "string",
  "params": {
    "[key: string]": "unknown"
  }
}
"""
"""
DAG格式例子(2025/4/16):
{
  "task": "分析服务器cpu异常原因",
  "nodes": {
    "node_1": {
      "name": "node_1",
      "agent_type": "系统感知Agent",
      "describe": "获取系统cpu状态",
      "edges": {
        "edge_1": {
          "type": "direct",
          "to_node": "node_2"
        },
        "edge_2": {
          "type": "conditional",
          "judge_function": "function_1",
          "to_node": {
            "yes": "node_3",
            "no": "node_4"
          }
        },
        "edge_3": {
          "type": "direct",
          "to_node": "node_4"
        }
      }
    },
    "node_2": {
      "name": "node_2",
      "agent_type": "异常分析Agent",
      "describe": "分析系统cpu状态诊断异常原因",
      "edges": {
        "edge_1": {
          "type": "direct",
          "to_node": "node_3"
        }
      }
    },
    "node_3": {
      "name": "node_3",
      "agent_type": "策略规划Agent",
      "describe": "根据cpu异常诊断结果规划解决方案并选择可执行操作",
      "edges": {
        "edge_1": {
          "type": "direct",
          "to_node": "node_4"
        },
        "edge_2": {
          "type": "direct",
          "to_node": "node_5"
        },
        "edge_3": {
          "type": "direct",
          "to_node": "node_6"
        }
      }
    },
    "node_4": {
      "name": "node_4",
      "agent_type": "操作执行Agent",
      "describe": "执行操作",
      "edges": {
        "edge_1": {
          "type": "direct",
          "to_node": "node_7"
        }
      }
    },
    "node_5": {
      "name": "node_5",
      "agent_type": "操作执行Agent",
      "describe": "执行操作",
      "edges": {
        "edge_1": {
          "type": "direct",
          "to_node": "node_7"
        }
      }
    },
    "node_6": {
      "name": "node_6",
      "agent_type": "操作执行Agent",
      "describe": "执行操作",
      "edges": {
        "edge_1": {
          "type": "direct",
          "to_node": "node_7"
        }
      }
    },
    "node_7": {
      "name": "node_7",
      "agent_type": "报告生成Agent",
      "describe": "根据操作执行结果结合异常分析结果汇总生成综合报告",
      "edges": {
        "edge_1": {
          "type": "direct",
          "to_node": "END"
        }
      }
    }
  }
}
"""
# 大模型初始化，这里使用的是api服务########################################################################################
llm_api = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                     api_key="sk-7e00adf9be1c431486a5c0a78eff1043")


# langgraph基础组件定义（类/函数等）########################################################################################
def concat(original: list, new: list) -> list:
    return original + new


def concat2(original: int, new: int) -> int:
    return new + original


class ChatState(TypedDict):
    messages: Annotated[list, concat]
    flag: Annotated[int, concat2]
    need_command: int
    query: str
    status: str  # "normal"/"error"


# agent相关函数#######################################################################################################
def node_func(state: ChatState):
    print("测试缓冲节点......")
    return


async def agent_quest(agent_url, data):
    """
    采用异步通信处理http交互，  任务解析——其他agent——大模型api
    data:dict
    """
    async with httpx.AsyncClient() as client:
        try:
            data = json.dumps(data)
            response = await client.post(agent_url, json=data, timeout=100000000)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            # 处理B服务返回的错误
            return {"error": f"B service error: {exc.response.text}"}


async def system_perceive(state: ChatState):
    """
        节点执行函数，根据当前节点类型来调用分机部署的agent
        """
    # 要发送的数据JSON-RPC 2.0规范
    print("执行system_perceive......")
    data = {
        "jsonrpc": "2.0",
        "id": "string | number",
        "method": "string",
        "params": {
            "[key: string]": "unknown"
        }

    }
    system_perceive_url = 'http://localhost:5004/system_perceive'
    # 调用agent
    result = await agent_quest(system_perceive_url, data)
    return {"flag": 1, "messages": [result]}


async def command_run(state: ChatState):
    """
        节点执行函数，根据当前节点类型来调用分机部署的agent
        """
    # 要发送的数据JSON-RPC 2.0规范
    print("执行command_run......")
    data = {
        "jsonrpc": "2.0",
        "id": "string | number",
        "method": "string",
        "params": {
            "[key: string]": "unknown"
        },
        "task": "test_task"
    }
    command_run_url = 'http://localhost:5003/command_run'
    # 调用agent
    result = await agent_quest(command_run_url, data)
    return {"flag": 1, "messages": [result]}


async def exception_analysis(state: ChatState):
    """
        节点执行函数，根据当前节点类型来调用分机部署的agent
        """
    # 要发送的数据JSON-RPC 2.0规范
    print("执行exception_analysis......")
    data = {
        "jsonrpc": "2.0",
        "id": "string | number",
        "method": "string",
        "params": {
            "[key: string]": "unknown"
        },
        "task": "test_task"
    }
    exception_analysis_url = 'http://localhost:5001/exception_analysis'
    # 调用agent
    result = await agent_quest(exception_analysis_url, data)
    return {"flag": 1, "messages": [result]}


async def report_generate(state: ChatState):
    """
        节点执行函数，根据当前节点类型来调用分机部署的agent
        """
    # 要发送的数据JSON-RPC 2.0规范
    print("执行report_generate......")
    data = {
        "jsonrpc": "2.0",
        "id": "string | number",
        "method": "string",
        "params": {
            "[key: string]": "unknown",
            "agent": "report_generate",
            "query": state["query"]

        }
    }
    report_generate_url = 'http://localhost:5005/report_generate'
    # 调用agent
    result = await agent_quest(report_generate_url, data)
    return {"flag": 1, "messages": [result]}


async def strategy_plan(state: ChatState):
    """
        节点执行函数，根据当前节点类型来调用分机部署的agent
        """
    # 要发送的数据JSON-RPC 2.0规范
    print("执行strategy_plan......")
    data = {
        "jsonrpc": "2.0",
        "id": "string | number",
        "method": "string",
        "params": {
            "[key: string]": "unknown"
        },
        "task": "test_task"
    }
    strategy_plan_url = 'http://localhost:5002/strategy_plan'
    # 调用agent
    result = await agent_quest(strategy_plan_url, data)
    return {"flag": 1, "messages": [result]}


# agent调用索引，按照节点名称配置合适的节点执行函数
agent_dict = {"操作执行Agent": command_run, "异常分析Agent": exception_analysis, "报告反馈Agent": report_generate,
              "策略规划Agent": strategy_plan, "系统感知Agent": system_perceive}


def choose_dag_model():
    """
    选择DAG模式：
    | 1  | **线性链（Linear Chain）**         | 串行任务               | 采集 → 分析 → 报告       |
    | 2️| **并行结构（Parallel Fork）**      | 同时调多源 Agent       | 指标、日志、配置同时提取    |
    | 3️| **依赖合并结构（Join）**            | 汇总子任务结果          | 多源数据合并分析          |
    | 4️| **条件分支（Conditional Branch）** | 基于判断走不同路径       | 判断异常类型 → 调用不同策略 |
    | 5️| **容错/回退路径（Fallback）**       | 遇故障自动切换处理策略   | 执行失败 → 改为人工审核     |
    | 6️| **子图嵌套（Sub-DAG）**            | 模块化表达复杂子流程     | 调用一个诊断子流程 DAG     |
    """
    prompt = f"现在有六种DAG模式：abcdef, 用户的问题为：xxxx请选择合适的DGA模式"
    model = "model"

    return "DAG_model"


# 任务解析主类，封装代码主要执行逻辑
class TaskAnalysis:
    def __init__(self, llm, query):
        """
        {}
        """
        self.llm = llm
        self.user_input = ''
        self.graph = ''
        self.result = ''
        self.query = query
        self.dag_prompt3 = '''
        - Role: 智能系统执行链规划师
        - Background: 目前有以下Agent类型:
          系统感知Agent是专家智能体，角色有:数据分析工程师，数据库专家，日志数据库专家 ，系统状态专家。他可以获取多个服务器的系统状态
          异常分析Agent是专家智能体，负责的任务有：日志异常检测，时间序列异常检测。
          策略规划Agent是专家智能体，他可以结合RAG技术规划策略，默认先匹配规则，没有规则的再由LLM规划。他会将要执行的操作交付对应服务器上的操作执行Agent去执行。
          操作执行Agent是专家智能体，扮演运维工程师的工作，如果用户的需求可以通过执行命令行满足，他会根据需求模拟需要执行的命令操作并执行。每台服务器都会部署该Agent,命名格式为:操作执行Agent_对应服务器名称。
          报告反馈Agent最后会将执行结果反馈给用户。对于一些操作手册相关的用户提问,他会使用rag技术根据本地知识库辅助回答用户的问题
          用户需要根据特定问题生成一条可执行的DAG（执行链），这涉及到对不同智能体功能的理解以及如何将它们合理组合来解决问题。
          用户的问题可能涉及多种领域，需要规划师能够灵活地运用各个智能体的能力来构建执行链。
        - Profile: 你是一位精通智能系统架构设计和执行流程规划的专家，对智能体之间的协作机制有着深刻的理解，能够根据不同的任务需求，设计出高效的执行链。
        - Skills: 你具备系统分析能力、智能体协作规划能力、DAG图设计能力以及对各种智能体功能的深入理解，能够快速识别任务的关键点，并将其转化为可执行的流程图。
        - Goals: 根据用户的问题，选择可用的agent, 生成一条可执行的DAG图，明确各智能体之间的执行顺序和依赖关系，即使问题超出常规范围，也要提供一个合理的DAG图作为解决方案。
        - Constrains: 生成的DAG图必须符合智能体的功能和协作逻辑，确保每个节点的任务清晰且可执行。节点仅包含提供的agent。即使问题不符合常规的规划范围，也要生成DAG，不能以“不知道”作为回答。只回答DAG，不要给出说明等无关内容。
        - OutputFormat: 以DAG图的形式输出，格式为json。
        - Workflow:
          1. 分析用户问题，明确任务需求和目标。
          2. 根据任务需求，确定参与的智能体及其角色。
          3. 设计智能体之间的执行顺序和依赖关系，生成DAG图。
        - Examples:
          - 例子1：用户问题："分析NAVI, serve1, test_server三台服务器的gpu异常原因并执行解决方案"
            {
              "task": "分析NAVI, serve1, test_server三台服务器的gpu异常原因并执行解决方案",
              "nodes": {
                "node_1": {
                  "name": "node_1",
                  "agent_type": "系统感知Agent",
                  "describe": "获取NAVI, serve1, test_server三台服务器的系统状态",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_2"
                    }
                  }
                },
                "node_2": {
                  "name": "node_2",
                  "agent_type": "异常分析Agent",
                  "describe": "分析系统gpu状态诊断异常原因",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_3"
                    }
                  }
                },
                "node_3": {
                  "name": "node_3",
                  "agent_type": "策略规划Agent",
                  "describe": "根据gpu异常诊断结果规划解决方案并选择可执行的修复操作",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_4"
                    },
                    "edge_2": {
                      "type": "direct",
                      "to_node": "node_5"
                    },
                    "edge_3": {
                      "type": "direct",
                      "to_node": "node_6"
                    }
                  }
                },
                "node_4": {
                  "name": "node_4",
                  "agent_type": "操作执行Agent",
                  "describe": "对服务器NAVI执行修复操作",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_7"
                    }
                  }
                },
                "node_5": {
                  "name": "node_5",
                  "agent_type": "操作执行Agent",
                  "describe": "对服务器serve1执行修复操作",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_7"
                    }
                  }
                },
                "node_6": {
                  "name": "node_6",
                  "agent_type": "操作执行Agent",
                  "describe": "对服务器test_server执行修复操作",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_7"
                    }
                  }
                },
                "node_7": {
                  "name": "node_7",
                  "agent_type": "报告反馈Agent",
                  "describe": "根据操作执行结果结合异常分析结果汇总生成综合报告",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "END"
                    }
                  }
                }
              }
            }  
          - 例子2：用户问题："查看服务器的cpu异常原因"
            {
              "task": "查看服务器的cpu异常原因",
              "nodes": {
                "node_1": {
                  "name": "node_1",
                  "agent_type": "系统感知Agent",
                  "describe": "获取服务器的系统状态",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_2"
                    }
                  }
                },
                "node_2": {
                  "name": "node_2",
                  "agent_type": "异常分析Agent",
                  "describe": "分析系统cpu状态诊断异常原因",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "node_3"
                    }
                  }
                },
                "node_3": {
                  "name": "node_3",
                  "agent_type": "报告反馈Agent",
                  "describe": "根据操作执行结果结合异常分析结果汇总生成综合报告",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "END"
                    }
                  }
                }
              }
            } 
          - 例子3：用户问题："常用vi命令有哪些"
            {
              "task": "查找常用vi命令相关资料",
              "nodes": {
                "node_1": {
                  "name": "node_1",
                  "agent_type": "报告反馈Agent",
                  "describe": "对于操作手册相关的知识类询问，结合rag技术直接回答用户的问题",
                  "edges": {
                    "edge_1": {
                      "type": "direct",
                      "to_node": "END"
                    }
                  }
                }
              }
            }
        - Initialization: 在第一次对话中，请给出以下问题的DAG:'''
        self.generate_langgraph_dag(query)

    def process_input(self):
        """
        处理输入
        """
        result = self.user_input
        return result

    def llm_generate_dag(self, task):
        """
        任务需求交付大模型生成DAG执行链
        """
        text = self.dag_prompt3 + task
        completion = self.llm.chat.completions.create(
            model = "qwen-turbo-2024-11-01", 
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': text}
            ],
            temperature = 0.6
        )

        content = completion.model_dump_json()
        content_dict = json.loads(content)

        raw_text = content_dict['choices'][0]['message']['content']

        # print(raw_text)
        if raw_text.startswith("无法解析任务"):
            result = "unresolved"
        elif raw_text.startswith("{"):
            result = json.dumps(raw_text)
            result = json.loads(result)

        else:
            result = "error"
        self.result = result
        print(result)
        return result


    def check_node_conflict(self, node, workflow):
        """
        检查节点是否冲突
        """
        if node in workflow.nodes.keys():
            print(f"节点{node}已存在，跳过创建")
            return False
        return True

    def create_parallel_fork(self, workflow, start_node: str, parallel_task: list):
        """
        为当前langgraph构建并行结构DAG
        workflow:langgraph
        parallel_task:list,存放要并行执行的任务
        """

        def parallel_start(state: ChatState):
            """
            可扩展代码，增添其他功能
            """
            return

        def parallel_result(state: ChatState):
            """
            可扩展代码，增添其他功能
            """
            return

        workflow.add_node("parallel_start", parallel_start)
        workflow.add_node("parallel_result", parallel_result)  # 接收并行执行后的结果
        workflow.add_edge(start_node, "parallel_start")
        for node in parallel_task:
            if self.check_node_conflict(node, workflow):
                workflow.add_node(node, agent_dict[node])
                self.create_fallback_judge_branch(workflow, node, "fallback_node")
            workflow.add_edge("parallel_start", node)
            workflow.add_edge(node, "parallel_result")
        return "parallel_result"

    def create_tasks_join(self, workflow, joined_task: list):
        """
        为当前langgraph构建汇总子任务结果(多源数据合并分析)DAG
        workflow:langgraph
        joined_task:list,存放要合并分析的任务
        task:先前已经在langgraph构建好的子任务链的列表(DAG)
        """
        print("合并任务结果")

        def join_start():
            """
            可扩展代码，增添其他功能
            """
            return

        def join_result():
            """
            可扩展代码，增添其他功能
            """
            return

        workflow.add_node("join_start", join_start)
        workflow.add_node("join_result", join_result)
        for task in joined_task:
            workflow.add_edge(task, "join_start")

        workflow.add_edge("join_start", "join_result")

    def create_conditional_branch(self, workflow, pre_node, judge_function, out_dict: dict):
        # 设置条件边
        workflow.add_conditional_edges(
            pre_node,
            judge_function,
            out_dict,
        )

    def create_fallback_judge_branch(self, workflow, pre_node, fallback_node):
        # 设置异常判断边
        def fallback_judge(state: ChatState):
            status = state["status"]
            if status == "error":
                return False
            elif status == "normal":
                return True

        workflow.add_conditional_edges(
            pre_node,
            fallback_judge,
            # {"error": fallback_node, "normal": dag_next_node},
            {"error": fallback_node},
        )
        print(f"为节点{pre_node}创建异常判断分支")

    def create_fallback_node(self, workflow):
        """
        遇故障自动切换处理策略,
        设置一个独立的异常处理节点,每个节点运行后使用条件边判断异常再交付
        """

        def fall_back(state: ChatState):
            """
            异常处理函数，可扩展增添其他功能
            """
            print("异常处理节点")
            print("系统状态:", state["status"])
            return

        workflow.add_node("fallback_node", fall_back)
        workflow.add_edge("fallback_node", END)
        print("异常处理节点创建完成")

    def create_sub_dag(self, workflow, last_node, sub_task):
        """
        模块化表达子流程
        """

        def generate_sub_nodes_and_edges(task):
            """
            子任务生成DAG
            """
            dag_chain = self.llm_generate_dag(task)
            if dag_chain == "unresolved":
                print("任务无法解析")
                return False
            elif dag_chain == "error":
                print("任务解析异常")
                return False
            else:
                print("正在生成子图DAG数据结构########################")
                nodes = dag_chain.split("-")
                edges = {}
                for i in range(len(nodes) - 1):
                    current_node = nodes[i]
                    next_node = nodes[i + 1]
                    edges[current_node] = [next_node]
                edges[nodes[-1]] = []
                nodes = nodes
                edges = edges
                return {"nodes": nodes, "edges": edges}

        sub_dag = generate_sub_nodes_and_edges(sub_task)
        if not sub_dag:
            print("由于未成功解析任务，无法生成langgraph")
            return

        # 添加节点
        print("创建子图节点...")
        workflow.add_node("sub_task_start", node_func)
        workflow.add_edge(last_node, "sub_task_start")

        for node in sub_dag["nodes"]:
            if self.check_node_conflict(node, workflow):
                print("子图节点", node)

                workflow.add_node(node, agent_dict[node])
        workflow.add_node("sub_task_end", node_func)
        # 添加边
        print("创建子图执行路径...")
        workflow.add_edge("sub_task_start", next(iter(sub_dag["edges"])))
        for out_node in sub_dag["edges"].keys():
            if sub_dag["edges"][out_node]:
                print("子图路径：", out_node, "---", sub_dag["edges"][out_node][0])
                workflow.add_edge(out_node, sub_dag["edges"][out_node][0])
        # 添加出口
        workflow.add_edge(list(sub_dag["edges"].keys())[-1], "sub_task_end")
        workflow.add_edge("sub_task_end", END)

    def generate_langgraph_dag(self, task):
        """
        根据大模型生成的DAG链条生成DAG数据结构
        """
        dag = self.llm_generate_dag(task)
        if dag.startswith("无法解析任务"):
            result = "unresolved"
            print("任务无法解析")
        elif dag.startswith("{"):
            print("构建langgraph#############################")
            dag = json.loads(dag)
            workflow = StateGraph(ChatState)
            # 初始化异常处理节点
            self.create_fallback_node(workflow)
            # 添加节点
            print("创建节点##############")
            workflow.add_node("start", node_func)
            nodes = dag["nodes"]
            for node in nodes.keys():
                agent_type = nodes[node]["agent_type"]
                describe = nodes[node]["describe"]
                print(f"节点:{node}, 类型:{agent_type}, 功能描述:{describe}")
                print("|")
                workflow.add_node(node, agent_dict[agent_type])
                self.create_fallback_judge_branch(workflow, node, "fallback_node")
            workflow.add_node("pre_end", node_func)

            # 添加边
            print("创建执行路径###########")
            workflow.set_entry_point("start")
            print("入口:", list(nodes.keys())[0])
            workflow.add_edge("start", list(nodes.keys())[0])
            for node in nodes.keys():
                edges = nodes[node]["edges"]
                for edge in edges.keys():
                    edge_type = edges[edge]["type"]
                    to_node = edges[edge]["to_node"]
                    if to_node == "END":
                        # 将最后一个节点与结束节点连接
                        print("出口:", node)
                        workflow.add_edge(node, "pre_end")
                        workflow.add_edge("pre_end", END)
                    else:
                        workflow.add_edge(node, to_node)

            # 编译图
            graph = workflow.compile(checkpointer=MemorySaver())
            self.graph = graph
            print("langgraph编译完成##########################")
        else:
            result = "error"
            print("任务解析异常")

    def draw_graph(self):
        self.graph.get_graph().print_ascii()

    async def run_graph(self):
        """
        运行graph,执行DAG链`
        """
        if self.result == "unresolved" or self.result == "error":
            print("由于未成功解析任务，无法执行langgraph")
            return
        # 执行图
        # 配置参数(线程id)
        config = {"configurable": {"thread_id": "1"}}
        prompt = "This is a prompt"
        output = await self.graph.ainvoke(
            {"flag": 0, "need_command": 0, "query": self.query},
            config=config
        )
        result = [output["messages"], output["flag"]]
        print("DAG执行结果")
        for message in output["messages"]:
            print(message)
        return result
        # generated_text = output["messages"][-1].content
        # print(output["need_command"], "---", "第", output["flag"], "轮", "Generated text:", generated_text)






'''
#################################################------本地代码-------######################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
#################################################------路由设置-------######################################################################################
'''

# 创建简易服务器
app = FastAPI()

# 允许外域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
def root():
    return {"Hello": "This is the root path"}


@app.post("/task_analysis")
async def task_analysis(request: Request) -> Response:
    request_dict = await request.json()
    request_dict = json.loads(request_dict)
    print(request_dict)
    query = "常用的vi指令有哪些"
    task_analysis = TaskAnalysis(llm_api, query)
    task_analysis.draw_graph()
    answer = await task_analysis.run_graph()
    return JSONResponse(answer)



@app.get("/test")
async def test_routine() -> Response:
    task_analysis = TaskAnalysis(llm_api, "我想了解服务器的系统异常原因")
    task_analysis.draw_graph()
    answer = await task_analysis.run_graph()
    return JSONResponse(answer)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)



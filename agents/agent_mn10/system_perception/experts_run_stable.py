import os
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import yaml
from pathlib import Path
from langgraph.graph import StateGraph, END

from agentsAPI import AgentState
from agent_mn10.system_perception.experts.state_expert_stable import StateExpert
from agent_mn10.system_perception.experts.tsdb_expert_v1 import TSdbExpert
from agent_mn10.system_perception.experts.logdb_expert import LogdbExpert


# 配置路径
agents_dir = Path(__file__).resolve().parents[2]
config_path = agents_dir / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


def route_expert(state: AgentState) -> AgentState:
    expert = state.get("expert", "")
    if expert == "state":
        return {**state, "next_node": "state_expert"}
    elif expert == "tsdb":
        return {**state, "next_node": "tsdb_expert"}
    elif expert == "logdb":
        return {**state, "next_node": "logdb_expert"}
    else:
        raise ValueError(f"Unknown expert: {expert}")


# 每个 Expert 节点的执行逻辑
async def state_expert(state: AgentState) -> AgentState:
    try:
        query = state["sub_task"]
        expert_config = config.get('sandbox', {})
        expert = StateExpert(expert_config)
        result = await expert.get_system_state(query)

        # 保证返回结构为 AgentState，result 为 dict
        return {
            **state,
            "status": result.get("status", "error"),
            "result": result.get("result", {}),
        }

    except Exception as e:
        return {
            **state,
            "status": "error",
            "error_code": 500,
            "error_message": str(e),
            "result": {"message": str(e)},
        }


def tsdb_expert(state: AgentState) -> AgentState:
    try:
        query = state["sub_task"]
        expert_config = config.get('prometheus', {})
        expert = TSdbExpert(expert_config)
        result = expert.get_tsdb_state(query)

        return {
            **state,
            "status": result.get("status", "error"),
            "result": result.get("result", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "error_code": 500,
            "error_message": str(e),
        }


def logdb_expert(state: AgentState) -> AgentState:
    try:
        query = state["sub_task"]
        expert_config = config.get('elasticsearch', {})
        expert = LogdbExpert(expert_config)
        result = expert.get_logdb_state(query)

        return {
            **state,
            "status": result.get("status", "error"),
            "result": result.get("result", {}),
        }

    except Exception as e:
        return {
            "status": "error",
            "error_code": 500,
            "error_message": str(e),
        }
    

async def state_expert_stream(state: AgentState):
    """
    流式版本的 state_expert，逐步返回执行结果
    """
    try:
        query = state["sub_task"]
        expert_config = config.get('sandbox', {})
        expert = StateExpert(expert_config)
        
        # 使用流式方法
        async for progress in expert.get_system_state_stream(query):
            yield {
                **state,
                **progress
            }

    except Exception as e:
        yield {
            **state,
            "status": "error",
            "error_code": 500,
            "error_message": str(e),
            "result": {"message": str(e)},
        }


async def tsdb_expert_stream(state: AgentState):
    """
    流式版本的 tsdb_expert，逐步返回执行结果
    """
    try:
        query = state["sub_task"]
        expert_config = config.get('prometheus', {})
        expert = TSdbExpert(expert_config)

        # 使用流式方法
        async for progress in expert.get_tsdb_state_stream(query):
            yield {
                **state,
                **progress
            }

    except Exception as e:
        yield {
            **state,
            "status": "error",
            "error_code": 500,
            "error_message": str(e),
            "result": {"message": str(e)},
        }


async def logdb_expert_stream(state: AgentState):
    """
    流式版本的 logdb_expert，逐步返回执行结果
    """
    try:
        query = state["sub_task"]
        expert_config = config.get('elasticsearch', {})
        expert = LogdbExpert(expert_config)

        # 使用流式方法
        async for progress in expert.get_logdb_state_stream(query):
            yield {
                **state,
                **progress
            }

    except Exception as e:
        yield {
            **state,
            "status": "error",
            "error_code": 500,
            "error_message": str(e),
            "result": {"message": str(e)},
        }


# 构建图
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("route", route_expert)
    workflow.add_node("state_expert", state_expert)
    workflow.add_node("tsdb_expert", tsdb_expert)
    workflow.add_node("logdb_expert", logdb_expert)

    workflow.add_conditional_edges(
        "route",
        lambda state: state["next_node"],
        {
            "state_expert": "state_expert",
            "tsdb_expert": "tsdb_expert",
            "logdb_expert": "logdb_expert",
        }
    )

    workflow.add_edge("state_expert", END)
    workflow.add_edge("tsdb_expert", END)
    workflow.add_edge("logdb_expert", END)

    workflow.set_entry_point("route")
    return workflow.compile()


# 构建流式图
def build_stream_graph():
    """
    构建支持流式输出的图
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("route", route_expert)
    workflow.add_node("state_expert_stream", state_expert_stream)
    workflow.add_node("tsdb_expert_stream", tsdb_expert_stream)
    workflow.add_node("logdb_expert_stream", logdb_expert_stream)

    workflow.add_conditional_edges(
        "route",
        lambda state: state["next_node"],
        {
            "state_expert": "state_expert_stream",
            "tsdb_expert": "tsdb_expert",
            "logdb_expert": "logdb_expert",
        }
    )

    workflow.add_edge("state_expert_stream", END)
    workflow.add_edge("tsdb_expert", END)
    workflow.add_edge("logdb_expert", END)

    workflow.set_entry_point("route")
    return workflow.compile()

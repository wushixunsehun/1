from typing import TypedDict, Annotated

# langgraph基础组件定义（类/函数等）
def concat(original: list[dict], new: list[dict]) -> list[dict]:
    return original + new


def override(old: str, new: str) -> str:
    return old if not new else new


def keep(old: str, new: str) -> str:
    return new if not old else old


def merge_result(old: dict, new: dict) -> dict:
    merged = dict(old or {})
    merged.update(new)
    return merged


def reduce_status(old: str, new: str) -> str:
    # 任何一个节点报 error，则整体视为 error
    return "error" if "error" in {old, new} else new


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], concat]    # 可用于存储消息记忆的状态

    query: Annotated[str, keep]    # 用户当前查询
    result: Annotated[dict, merge_result]    # 当前查询的结果

    status: Annotated[str,  reduce_status]    # "success"/"error"
    error_code: int    # 错误码
    error_message: str    # 错误信息

    hostname: Annotated[str, override]    # 目标对象
    sub_task: Annotated[str, override]
    expert: Annotated[str, override]

    confirm: dict[str, str]  # 二次确认参数，格式为 {host: Y/N}
    cmds_for_confirm: dict  # 首次生成的命令，供二次确认时直接使用
    session_id: Annotated[str, override]  # 会话唯一标识，用于DAG恢复和参数补充

    _msg_keys: set

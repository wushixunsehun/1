import os, re, yaml
import grpc, json, inspect
from openai import OpenAI
from agent_state import AgentState
from typing import Callable, Optional
from rag_grpc import rag_pb2 as rag_pb2
from rag_grpc import rag_pb2_grpc as rag_pb2_grpc

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


llm_conf = config['llm']
OPENAI_LLM_CLIENT = OpenAI(
    base_url = llm_conf["base_url"],
    api_key = llm_conf["api_key"],
)
llm_model = llm_conf['model']


emb_conf = config['rag']
OPENAI_EMB_CLIENT = OpenAI(
    base_url = emb_conf["base_url"],
    api_key = emb_conf["api_key"],
)
emb_model = emb_conf['embedding_model']


TokenCb = Callable[[str], None]


def get_rag_rpc(query: str, source: list):
    """
    通过 gRPC 调用 RAG 服务获取上下文
    """
    with grpc.insecure_channel('a6000-G5500-V6:5413') as channel:
        stub = rag_pb2_grpc.RagServiceStub(channel)
        req = rag_pb2.RagRequest(query=query, source=source)
        resp = stub.GetRagS2C(req)
        return resp.context


def get_rag_rpc_only_content(query: str):
    """
    通过 gRPC 调用 RAG 服务获取上下文
    """
    with grpc.insecure_channel('a6000-G5500-V6:5413') as channel:
        stub = rag_pb2_grpc.RagServiceStub(channel)
        req = rag_pb2.RagRequest_v2(query=query)
        resp = stub.GetRagContent(req)
        return resp.context


def strip_think(text: str) -> str:
    """删除 <think>…</think> 块"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def query_embedding(text: list[str]) -> str:
    """
    输入 text 到嵌入模型，获取嵌入，走 OpenAI 接口
    """
    completion = OPENAI_EMB_CLIENT.embeddings.create(
        model = emb_model,
        input = text,
        encoding_format = "float"
    )

    rep = json.loads(completion.model_dump_json())['data'][0]['embedding']
    return rep


def query_llm(prompt: str, *, stream: bool = False, enable_thinking: bool = False, on_token: Optional[TokenCb] = None, **kwargs) -> str:
    """
    输入 prompt，询问 LLM，获取回复，走 OpenAI 接口
    - stream=True 时，返回 async generator，可 async for 消费流式 token。
    - on_token 支持 async 回调，便于上层异步处理 token。
    """
    try:
        # --- 调用 LLM ---
        response = OPENAI_LLM_CLIENT.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "你是一名智能助理。"},
                {"role": "user", "content": prompt},
            ],
            stream=stream,
            extra_body={"enable_thinking": enable_thinking},
            **kwargs
        )

        # --- 流式输出 ---
        if stream:
            async def _gen():
                for chunk in response:
                    token_raw = chunk.choices[0].delta.content or ""
                    token_vis = token_raw if enable_thinking else strip_think(token_raw)
                    if token_vis:
                        if on_token:
                            if inspect.iscoroutinefunction(on_token):
                                await on_token(token_vis)
                            else:
                                on_token(token_vis)
                        yield token_vis
            return _gen()
        
        # --- 非流式 ---
        full_raw = response.choices[0].message.content
        return full_raw if enable_thinking else strip_think(full_raw)
    except Exception as e:
        code = getattr(e, "status_code", "N/A")
        raise RuntimeError(f"LLM error {code}: {e}") from e

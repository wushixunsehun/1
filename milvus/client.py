import json
import time
import yaml
import httpx
import logging
from openai import OpenAI


logging.basicConfig(
    filename = "llm_query_test.log",
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


config_path = 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

llm_conf = config['llm']
llm = OpenAI(
    base_url = llm_conf['base_url'],
    api_key = llm_conf['api_key'],
)


def query_llm(prompt: str) -> str:
    completion = llm.chat.completions.create(
        model = llm_conf['model'],
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
    )

    content = completion.model_dump_json()
    content_dict = json.loads(content)

    response = content_dict['choices'][0]['message']['content']
    return response


def query_llm_stream(prompt: str) -> str:
    completion = llm.chat.completions.create(
        model = llm_conf['model'],
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature = 0.3,
        max_tokens = 4096,
        top_p = 0.8,
        presence_penalty = 1.5,
        frequency_penalty = 1.5,
        stream = True
    )

    for chunk in completion:
        if delta := chunk.choices[0].delta:
            print(delta.content, end="", flush=True)


def main() -> None:
    query = 'TH-3F系统如何安装和使用可视化工具NCL？'

    # 直接访问，最终返回整个答案

    logger.info(f"输入查询：{query}")
    response = query_llm(query)
    logger.info(f"模型输出：\n{response}")


    # 流式输出，测试

    # print(f'输入查询：{query}')
    # query_llm_stream(query)


if __name__ == '__main__':
    start_main_time = time.perf_counter()
    main()
    end_main_time = time.perf_counter()


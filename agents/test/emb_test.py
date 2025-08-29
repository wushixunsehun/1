import json
from openai import OpenAI
import time


client = OpenAI(
    base_url = "http://a6000-G5500-V6:5415/v1",
    api_key = "EMPTY",
)


def query_emb(text: list[str]) -> str:
    """
    输入 prompt，询问 LLM
    """
    completion = client.embeddings.create(
        model = "Alibaba-NLP/gte-multilingual-base",
        input = text,
        encoding_format = "float"
    )

    rep = json.loads(completion.model_dump_json())['data'][0]['embedding']
    return rep


def main() -> None:
    text = "中国的首都是北京"
    response = query_emb([text])
    # print(response)


if __name__ == '__main__':
    start_main_time = time.perf_counter()
    main()
    end_main_time = time.perf_counter()
    print(f'Run time: {end_main_time - start_main_time:.4f}s')


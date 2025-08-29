import json
from openai import OpenAI
from transformers import T5Tokenizer, T5ForConditionalGeneration


class SummaryGenerator:
    def __init__(self, summary_model_name, config):
        self.summary_model_name = summary_model_name
        self.device = config['device']

        if self.summary_model_name  == 't5':
            model_name = 'utrobinmv/t5_summary_en_ru_zh_base_2048'
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)

            self.model.eval()
            self.model.to(self.device)
        elif self.summary_model_name[:3] == 'qwq':
            self.client = OpenAI(
                api_key = config['api_key'],
                base_url = config['base_url'],
            )
            self.model = config['model']


    def gen_summary_t5(self, chunk: str) -> str:
        prefix = 'summary: '
        src_text = prefix + chunk

        input_ids = self.tokenizer(src_text, return_tensors='pt')
        generated_tokens = self.model.generate(**input_ids.to(self.device))
        summary = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return summary[0]


    def gen_summary_qwq(self, chunk: str) -> str:
        PROMPT_TEMPLATE = """为以下文本进行摘要总结：{chunk}，尽可能涵盖大部分内容，字数不超过 300。
        无需返回除纯文本以外的其他任何内容.
        """

        prompt = PROMPT_TEMPLATE.format(chunk=chunk)
        completion = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
        )

        content = completion.model_dump_json()
        summary_dict = json.loads(content)

        summary = summary_dict['choices'][0]['message']['content']
        return summary


    def gen_summary(self, chunk: str) -> str:
        if self.summary_model_name == 't5':
            return self.gen_summary_t5(chunk)
        elif self.summary_model_name[:3] == 'qwq':
            return self.gen_summary_qwq(chunk)
        else:
            raise ValueError(f"Unknown method: {self.summary_model_name}")


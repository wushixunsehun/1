import re
import os
import time
import jieba
import requests
import tempfile
import mimetypes
import pytesseract
from PIL import Image
from tqdm import tqdm
from sentence_transformers import util

import public_function as pf

class MdProcessor():
    """处理 Markdown 文件的类"""
    def __init__(self, embedding_model, summary_generator):
        self.embedding_model = embedding_model
        self.summary_generator = summary_generator

    def load_and_clean_md(self, file_path):
        """清洗开头信息，返回正文内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]
            lines = [line for line in lines if line]

        return lines

        # clean_start = False
        # cleaned = []

        # author_pattern = re.compile(r'^\*\*作者\*\*:')
        # label_pattern = re.compile(r'^\*\*标签\*\*:')
        
        # if lines:
        #     cleaned.append(lines[0].strip('\n'))

        # for line in lines[1:]:
        #     line = line.strip('\n')

        #     if not line.strip(): # 空行
        #         continue
            
        #     if label_pattern.match(line):
        #         cleaned.append(line)
        #         continue

        #     if not clean_start:
        #         if author_pattern.match(line):
        #             clean_start = True
        #         continue

        #     if clean_start:
        #         cleaned.append(line)

        # return cleaned
    
    def parse_lines_to_blocks(self, lines, base_dir):
        """解析 Markdown 内容结构，返回 [结构化文本列表]"""
        blocks = []
        in_code = False
        in_table = False
        code_block = []
        table_block = []
        i = 0

        for i in range(len(lines)):
            line = lines[i].strip()

            # 代码块处理
            if line.startswith("```"):
                if in_code:
                    blocks.append("<!--代码块开始-->\n" + '\n'.join(code_block) + "\n<!--代码块结束-->")
                    code_block.clear()
                in_code = not in_code
                continue

            if in_code:
                code_block.append(line)
                continue

            # 表格处理
            if '|' in line and re.match(r'^\s*\|.*\|\s*$', line):
                cleaned_line = '|'.join(cell.strip() for cell in line.strip().split('|'))
                
                # 新判断：是否是 markdown 表格的格式，而非 ASCII 图形
                pipe_count = cleaned_line.count('|')
                word_ratio = len(re.findall(r'\w', cleaned_line)) / (len(cleaned_line) + 1e-5)
                looks_like_real_table = pipe_count >= 2 and word_ratio > 0.2

                if looks_like_real_table:
                    in_table = True
                    # 忽略分隔线（表头后）
                    if len(table_block) == 1 and re.match(r'^\s*[:-]+[-| :]*$', cleaned_line):

                        continue

                    table_block.append(cleaned_line)
                    continue
                else:
                    # 伪表格按正文处理
                    blocks.append("<!--正文开始-->\n" + line)
                    continue

            elif in_table and not re.match(r'^\s*\|.*\|\s*$', line):
                blocks.append("<!--表格开始-->\n" + '\n'.join(table_block) + "\n<!--表格结束-->")
                table_block.clear()
                in_table = False

            if in_table:
                cleaned_line = '|'.join(cell.strip() for cell in line.strip().split('|'))
                table_block.append(cleaned_line)
                continue

            # 标题处理
            if re.match(r'^#{1,6} ', line):
                blocks.append("<!--标题开始-->\n" + line.lstrip('# ').strip())
                continue

            # 列表块
            if re.match(r'^(\d+\.\s+|- |\* )', line):  # 列表开头
                list_block = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    # 判断是否为子项（缩进）、空行、或者与前项逻辑连续
                    if re.match(r'^\s+[\-\*\d]', next_line) or re.match(r'^\s*$', next_line):
                        list_block.append(next_line.strip())
                        i += 1
                    elif re.match(r'^\s+', next_line):  # 缩进正文说明也并入
                        list_block.append(next_line.strip())
                        i += 1
                    else:
                        break
                blocks.append("<!--列表开始-->\n" + '\n'.join(list_block) + "\n<!--列表结束-->")
                continue

            # 图片（支持本地+网络）
            img_match = re.match(r'!\[.*?\]\((.*?)\)', line)
            if img_match:
                img_path = img_match.group(1)
                text = self.extract_text_from_image(img_path, base_dir)
                blocks.append("<!--图片开始-->\n" + text + "\n<!--图片结束-->")
                continue

            # 正文
            blocks.append("<!--正文开始-->\n" + line)

        # 处理未结束的表格
        if table_block:
            blocks.append("<!--表格开始-->\n" + '\n'.join(table_block) + "\n<!--表格结束-->")

        return blocks
    
    def extract_text_from_image(self, path, base_dir):
        try:
            if path.startswith("http://") or path.startswith("https://"):
                headers = {'User-Agent': 'Mozilla/5.0'}
                resp = requests.get(path, timeout=10, headers=headers)
                if resp.status_code == 200:
                    content_type = resp.headers.get('Content-Type', '')
                    extension = mimetypes.guess_extension(content_type.split(';')[0]) or '.png'
                    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name
                else:
                    return "[无法下载图片]"
            else:
                tmp_path = os.path.join(base_dir, path)
                if not os.path.exists(tmp_path):
                    return "[图片不存在]"

            with Image.open(tmp_path) as img:
                return pytesseract.image_to_string(img, lang='chi_sim+eng').strip()
        except Exception as e:
            return f"[图片识别失败: {str(e)}]"
    
    def clean_noise(self, text):
        # 去除添加的 HTML 注释标记
        text = re.sub(r'<!--.*?开始-->', '', text)
        text = re.sub(r'<!--.*?结束-->', '', text)

        # 去除 markdown 表格分隔线，如 |----|----|
        text = re.sub(r'^\s*\|?[\s:\-|]+\|?\s*$', '', text, flags=re.MULTILINE)

        text = re.sub(r'[-＝=~—_⋯·•]{2,}', '', text, flags=re.MULTILINE)

        # 合并多个空行为一个
        text = re.sub(r'\n\s*\n+', '\n', text)

        # 去除每一行首尾的空格
        lines = text.split('\n')
        cleaned_text = '\n'.join(line.strip() for line in lines)

        return cleaned_text.strip()
    
    def tokenize_text_jieba(self, text):
        """
        使用 jieba 对中英文混合文本进行分词：
        - 中文用 jieba
        - 英文/数字用正则保留原格式
        """
        # jieba 对整体做一次初步分词
        words = jieba.lcut(text)
        tokens = []
        for word in words:
            # 英文、数字、IP地址、文件名等直接识别
            if re.match(r'^[a-zA-Z0-9_\-./\\:]+$', word):
                tokens.append(word)
            else:
                tokens.append(word)

        return tokens

    def slide_split(self, text, max_len=500, overlap=50):
        """基于 jieba 中文分词 + 英文正则 分块"""
        cleaned_text = self.clean_noise(text)

        tokens = self.tokenize_text_jieba(cleaned_text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_len, len(tokens))
            chunk = ''.join(tokens[start: end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(tokens):
                break
            start += (max_len - overlap)

        return chunks

    def cluster_by_similarity(self, chunks, threshold=0.65, max_group_size=3):
        """基于相似度合并为章节，并生成摘要"""
        merged_sections = []
        current_group = []

        i = 0
        while i < len(chunks):
            current = chunks[i]
            prev_start = max(0, i - 2)
            prev_neighbors = chunks[prev_start: i]
            next_end = min(len(chunks), i + 3)
            next_neighbors = chunks[i + 1: next_end]

            # 前两块与后两块作为当前块的邻居
            neighbors = prev_neighbors + next_neighbors

            if neighbors:
                emb_current = self.embedding_model.encode(current, normalize_embeddings=True)
                emb_neighbors = self.embedding_model.encode(neighbors, normalize_embeddings=True)
                sim_scores = self.embedding_model.similarity(emb_current, emb_neighbors)
                avg_sim = sim_scores.mean().item()

                if avg_sim > threshold and len(current_group) < max_group_size:
                    current_group.append(current)
                else:
                    # 若当前已有聚合块，先处理它
                    if current_group:
                        full_text = '。'.join(current_group)
                        summary = self.summary_generator.gen_summary(full_text)
                        time.sleep(0.5)
                        merged_sections.append({
                            'summary': summary,
                            'chunks': current_group.copy()
                        })
                        current_group.clear()
                    current_group.append(current)
            else:
                current_group.append(current)
            
            i += 1

        # 处理最后一组
        if current_group:
            full_text = '。'.join(current_group)
            summary = self.summary_generator.gen_summary(full_text)
            merged_sections.append({
                'summary': summary,
                'chunks': current_group.copy()
            })

        return merged_sections
    
    def read_md(self, file_path: str, dataset) -> list:
        """入口：读取 markdown 文件，处理后返回 [summary + chunks] 列表"""
        # 创建文件保存路径
        dataset_path = f'preprocess_data_files/{dataset}/'
        os.makedirs(dataset_path, exist_ok=True)

        content_save_directory = dataset_path + 'content/'
        summary_save_directory = dataset_path + 'summary_500/'
        os.makedirs(content_save_directory, exist_ok=True)
        os.makedirs(summary_save_directory, exist_ok=True)

        file_name = file_path.split('/')[-1:][0].replace('.md', '.txt')
        content_save_path = os.path.join(content_save_directory, file_name)
        summary_save_path = os.path.join(summary_save_directory, file_name)

        # if os.path.exists(content_save_path) and os.path.exists(summary_save_path):
        #     return pf.read_data_from_txt(summary_save_path)
        
        # 1. 已有文档，直接读取
        if os.path.exists(content_save_path):
            blocks = pf.read_data_from_txt(content_save_path)
        else:
            # 2. 否则解析 Markdown 内容结构
            lines = self.load_and_clean_md(file_path)
            blocks = self.parse_lines_to_blocks(lines, base_dir=os.path.dirname(file_path))
            pf.write_data_to_txt(blocks, content_save_path)

        # 3. 滑动分块
        text = '\n'.join(blocks)
        chunks = self.slide_split(text)

        # 4. 相似度合并
        clustered_sections = self.cluster_by_similarity(chunks)
        pf.write_data_to_txt(clustered_sections, summary_save_path)

        return clustered_sections

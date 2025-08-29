import io
import re
import os
import time
import jieba
import pytesseract
from PIL import Image
from tqdm import tqdm
from sentence_transformers import util

from docx import Document  # python-docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

import public_function as pf


class DocxProcessor:
    def __init__(self, embedding_model, summary_generator):
        self.embedding_model = embedding_model
        self.summary_generator = summary_generator


    def iter_block_items(self, parent_doc):
        """遍历段落和表格（图片由段落中识别）"""
        for child in parent_doc.element.body.iterchildren():
            if child.tag == qn('w:p'):
                yield Paragraph(child, parent_doc)
            elif child.tag == qn('w:tbl'):
                yield Table(child, parent_doc)


    def extract_images_from_paragraph(self, paragraph):
        """从段落中提取图片内容（按顺序）"""
        images = []
        for run in paragraph.runs:
            drawing = run._element.xpath('.//pic:pic')
            if drawing:
                # 查找 embed ID
                blip = run._element.xpath('.//a:blip')
                if blip:
                    embed = blip[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                    image_part = paragraph.part.related_parts.get(embed)
                    if image_part:
                        images.append(image_part.blob)

        return images


    def extract_text_from_image(self, image):
        return pytesseract.image_to_string(image, lang='chi_sim+eng')


    def extract_table_text(self, table):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        return '\n'.join(rows)


    def is_possible_toc(self, para: Paragraph) -> bool:
        text = para.text.strip()
        if re.findall(r'(第[\d一二三四五六七八九十百]+[章节篇])', text) and re.search(r'\.+\s*\d{1,3}$', text):
            return True
        if re.search(r'^\d+(\.\d+)+\s+.+\.+\s*\d{1,3}$', text):
            return True
        return False


    def clean_text(self, text):
        if not text:
            return ""
        # 去除制表符、全角空格、连续空格等
        text = text.replace('\t', '')
        text = text.replace('\u3000', ' ')  # 全角空格
        text = re.sub(r'[ ]{2,}', '', text)  # 连续空格
        return text.strip()


    def parse_docx(self, file_path):
        document = Document(file_path)
        content_blocks = []

        for block in tqdm(self.iter_block_items(document), desc='文档读取'):
            if isinstance(block, Paragraph):
                text = self.clean_text(block.text)
                if text and not self.is_possible_toc(block):
                    content_blocks.append(text)

                # 提取段落中的图片
                images = self.extract_images_from_paragraph(block)
                for img_blob in images:
                    try:
                        image = Image.open(io.BytesIO(img_blob))
                        ocr_text = self.clean_text(self.extract_text_from_image(image))
                        if ocr_text:
                            content_blocks.append(ocr_text)
                    except Exception as e:
                        print(f"[警告] 图片处理失败：{e}")

            elif isinstance(block, Table):
                table_text = self.clean_text(self.extract_table_text(block))
                if table_text:
                    content_blocks.append(table_text)

        return content_blocks


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
        tokens = self.tokenize_text_jieba(text)
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

        with tqdm(total=len(chunks), desc='相似度计算') as pbar:
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
                    # sim_scores = util.cos_sim(emb_current, emb_neighbors)
                    avg_sim = sim_scores.mean().item()

                    if avg_sim > threshold and len(current_group) < max_group_size:
                        current_group.append(current)
                    else:
                        # 若当前已有聚合块，先处理它
                        if current_group:
                            full_text = '。'.join(current_group)
                            summary = self.summary_generator.gen_summary(full_text)
                            time.sleep(1)
                            merged_sections.append({
                                'summary': summary,
                                'chunks': current_group.copy()
                            })
                            current_group.clear()
                        current_group.append(current)
                else:
                    current_group.append(current)

                pbar.update(1)
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


    def read_docx(self, file_path: str, dataset) -> list:
        """入口：读取 docx 文件，处理后返回 [summary + chunks] 列表"""
        # 创建文件保存路径
        dataset_path = f'preprocess_data_files/{dataset}/'
        os.makedirs(dataset_path, exist_ok=True)

        content_save_directory = dataset_path + 'content/'
        summary_save_directory = dataset_path + 'summary_500/'
        os.makedirs(content_save_directory, exist_ok=True)
        os.makedirs(summary_save_directory, exist_ok=True)

        file_name = file_path.split('/')[-1:][0].replace('.docx', '.txt')
        content_save_path = os.path.join(content_save_directory, file_name)
        summary_save_path = os.path.join(summary_save_directory, file_name)

        # if os.path.exists(content_save_path) and os.path.exists(summary_save_path):
        #     return pf.read_data_from_txt(summary_save_path)

        # 1. 读取文档，提取文本、图片、表格
        if os.path.exists(content_save_path):
            doc_contents = pf.read_data_from_txt(content_save_path)
        else:
            doc_contents = self.parse_docx(file_path)
            pf.write_data_to_txt(doc_contents, content_save_path)

        # 2. 滑动分块
        text = '\n'.join(doc_contents)
        chunks = self.slide_split(text)

        # 3. 相似度合并
        clustered_sections = self.cluster_by_similarity(chunks)
        pf.write_data_to_txt(clustered_sections, summary_save_path)

        return clustered_sections


import io
import re
import os
import time
import jieba
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from tqdm import tqdm

import public_function as pf

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class PdfProcessor:
    """处理 PDF 文件的类"""
    def __init__(self, embedding_model, summary_generator):
        self.embedding_model = embedding_model
        self.summary_generator = summary_generator

    def ocr_page(self, page):
        """将 PDF 页面转换为图像并进行 OCR 识别"""
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes('png')))
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        return text

    def is_toc_page(self, text):
        """判断是否为目录页"""
        lines = text.split('\n')
        strong_match = 0  # “第X章”、“附录”等明确章节编号
        number_heading_match = 0  # “1.2.3 标题”类结构
        valid_lines = 0

        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue
            valid_lines += 1

            # 强信号章节编号
            if re.match(r'^(第[一二三四五六七八九十百]+章|第\d+章|附录|[A-Z]\d*)(\s|:|：)', line):
                strong_match += 1

            # 弱信号编号，如 1.2.3 标题 / .3.2.1 标题
            if re.match(r'^\.?\d+(\.\d+){1,3}\s+[\u4e00-\u9fa5A-Za-z]', line):
                if not re.match(r'^\d+(\.\d+){1,3}$', line):  # 排除纯数字（IP/版本号）
                    number_heading_match += 1

        # 判定标准（保守策略）
        return (
            strong_match >= 3 or
            (number_heading_match >= 5 and valid_lines > 6)
        )

    def ocr_page_with_markers(self, page, page_index):
        """对单页进行 OCR"""
        text_result = ''

        # OCR 文本
        pix = page.get_pixmap(dpi=400)
        img = Image.open(io.BytesIO(pix.tobytes('png')))
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        text = text.strip()

        text_result += f'\n{text}'
        return text_result
    
    def clean_ocr_text(self, text: str) -> str:
        """清理 OCR 文本中的多余空格和空行"""
        # 去除每行两端空格，并替换多余空格为一个
        lines = [re.sub(r'\s+', ' ', line.strip()) for line in text.split('\n')]
        # 去除空行
        lines = [line for line in lines if line]
        return ''.join(lines)

    def extract_all_text(self, file_path):
        """整份 PDF 进行 OCR，跳过封面和目录，并插入图片/表格标记"""
        doc = fitz.open(file_path)
        full_text = []

        for i, page in enumerate(tqdm(doc, desc='文档读取')):
            if i == 0:
                continue  # 跳过封面

            text_preview = self.ocr_page_preview(page)  # 先粗提文本判断是否目录

            if self.is_toc_page(text_preview):
                continue

            page_text = self.ocr_page_with_markers(page, i)
            page_text = self.clean_ocr_text(page_text)

            full_text.append(page_text)

        return full_text

    def ocr_page_preview(self, page):
        """获取 OCR 文本预览（低分辨率）用于判断是否为目录"""
        pix = page.get_pixmap(dpi=170)
        img = Image.open(io.BytesIO(pix.tobytes('png')))
        return pytesseract.image_to_string(img, lang='chi_sim+eng').strip()

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

    def read_pdf(self, file_path: str, dataset) -> list:
        """入口：读取 pdf 文件，处理后返回 [summary + chunks] 列表"""
        # 创建文件保存路径
        dataset_path = f'preprocess_data_files/{dataset}/'
        os.makedirs(dataset_path, exist_ok=True)

        content_save_directory = dataset_path + 'content/'
        summary_save_directory = dataset_path + 'summary_500/'
        os.makedirs(content_save_directory, exist_ok=True)
        os.makedirs(summary_save_directory, exist_ok=True)

        file_name = file_path.split('/')[-1:][0].replace('.pdf', '.txt')
        content_save_path = os.path.join(content_save_directory, file_name)
        summary_save_path = os.path.join(summary_save_directory, file_name)

        # if os.path.exists(content_save_path) and os.path.exists(summary_save_path):
        #     return pf.read_data_from_txt(summary_save_path)
        
        # 1. OCR 全文本
        if os.path.exists(content_save_path):
            text = pf.read_data_from_txt(content_save_path)
        else:
            text = self.extract_all_text(file_path)
            pf.write_data_to_txt(text, content_save_path)

        # 2. 滑动分块
        full_text = "\n".join(text)
        chunks = self.slide_split(full_text)

        # 3. 相似度合并
        merged_sections = self.cluster_by_similarity(chunks)
        pf.write_data_to_txt(merged_sections, summary_save_path)

        return merged_sections

import os
from processor.PdfProcessor import PdfProcessor
from processor.MdProcessor import MdProcessor
from processor.DocxProcessor import DocxProcessor
from processor.ExcelProcessor import ExcelProcessor


def process_files(file_paths: list, dataset, embedding_model, summary_generator) -> list:
    """处理多个文件，返回分块后的文本列表"""
    docs = []
    item_cnt = 1

    pdf_processor = PdfProcessor(embedding_model, summary_generator)
    md_processor = MdProcessor(embedding_model, summary_generator)
    docx_processor = DocxProcessor(embedding_model, summary_generator)
    excel_processor = ExcelProcessor(embedding_model, summary_generator)

    # 轮询处理文件夹里的不同文档，并根据文档类型分块
    for file_path in file_paths:
        print(f'[{item_cnt}/{len(file_paths)}] 文件：{file_path}')
        if file_path.endswith('.pdf'):
            section_chunks = pdf_processor.read_pdf(file_path, dataset)
        elif file_path.endswith('.md'):
            section_chunks = md_processor.read_md(file_path, dataset)
        elif file_path.endswith('.docx'):
            section_chunks = docx_processor.read_docx(file_path, dataset)
        elif file_path.endswith('.xlsx'):
            section_chunks = excel_processor.read_excel(file_path, dataset)
        else:
            raise ValueError(f'不支持的文件类型: {file_path}')

        item_cnt += 1

        # 将文档摘要和内容加入总列表
        docs.extend(section_chunks)

    return docs


def read_docs(docs_file_path: str, dataset, embedding_model, summary_generator) -> list:
    """读取指定文件夹中的所有文档文件"""

    if not os.path.isdir(docs_file_path):
        raise ValueError('提供的路径不是一个文件夹')
    
    file_paths = []
    for file_name in sorted(os.listdir(docs_file_path)):
        file_path = os.path.join(docs_file_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(('.pdf', '.md', 'docx', 'xlsx')):
            file_paths.append(file_path)

    return process_files(file_paths, dataset, embedding_model, summary_generator)


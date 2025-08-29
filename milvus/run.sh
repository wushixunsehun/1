#!/bin/bash

# 定义数据集列表
datasets=("markdown" "pdf" "docx" "html2md" "excel")

# 遍历数据集并执行脚本
for dataset in "${datasets[@]}"; do
    python milvus_demo.py --dataset "$dataset"
done


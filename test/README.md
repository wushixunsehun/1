Author: yuaw

# NSCC-test

## 📖 索引
1. [项目简介](#-项目简介)
2. [如何使用](#-如何使用)

## 📝 项目简介
这是使用 Python 复现的五种基线测试方法及它们在问题集下的性能测试结果。

### 📘 项目分支结构
```
├── answer/    # 五种基线方法的答案
│   ├── answer_lightrag/
|   ├── answer_metagpt/
|   ├── answer_naiverag/
|   ├── answer_react/
|   ├── answer_selfrag/
|
├── figures/   # 存放测试结果的图片
│   ├── Metagpt-1.png    # 基线方法对应图片
│   ├── Metagpt-2.png
│   ├── ...
|   ├── data_meta.m    # 画图文件 
│   ├── ...
│   
├── file_response/    # 数据库中返回的和问题相关的内容
│   ├── file_response_agent/
│   ├── file_response_rags/
│
├── lightrag_code/    # lightrag方法
│   ├── demo/
|       ├── dickens/  # 存放索引文件
|       ├── ...
|       ├── lightrag_openai_main.py    # 输入问题，运行lightrag代码，获取答案
|   ├── LightRAG-main.zip   # lightrag源代码
│   ├── requirements.txt/   # 环境依赖文件，主要pip install lightRAG
│
├── metagpt_code/    # metagpt方法
│   ├── demo/
|       ├── config/  # 存放API配置文件
|       ├── ...
|       ├── multi_qwendb2_tokentime.py    # 输入问题，运行metagpt代码，获取答案
|   ├── MetaGPT-main.zip   # metagpt源代码
│   ├── requirements.txt/   # 环境依赖文件，主要pip install metagpt
│ 
├── metric/    # 测试结果日志及指标
│   ├── metric_lightrag_log/    # 测试lightrag的中间结果日志
│   ├── metric_metagpt_log/    # 测试metagpt的中间结果日志
│   ├── metric_naiverag_log/    # 测试naiverag的中间结果日志
│   ├── metric_react_log/    # 测试react的中间结果日志
│   ├── metric_selfrag_log/    # 测试selfrag的中间结果日志
|   ├── test_code/
|       ├── deepeval-main.zip/  # deepeval源代码
|       ├── ...
|       ├── test_example_react.py    # 测试react方法输出结果质量的脚本
│   ├── metric_data.xls    # 测试结果数据对比
│ 
├── naiverag_code/    # naiverag方法
│   ├── rag_tokentime_main.py    # 输入问题，运行naiverag代码，获取答案
|   ├── ...
|   ├── requirements.txt/   # 环境依赖文件
│ 
├── question_dataset/    # 问题集
│   ├── 问题集.pdf
│ 
├── react_code/    # react方法
│   ├── react_tokentime_main.py    # 输入问题，运行react代码，获取答案
|   ├── ...
|   ├── requirements.txt/   # 环境依赖文件
│ 
├── selfrag_code/    # selfrag方法
│   ├── selfrag_tokentime_main.py    # 输入问题，运行naiverag代码，获取答案
|   ├── ...
|   ├── requirements.txt/   # 环境依赖文件
│ 
├── ... 
```

## ❓ 如何使用
✅ 运行基线方法时，确保安装对应的环境依赖，并配置API和embedding model/开启服务器上的qwq30b.service和embedding_api.service服务。
```bash
systemctl status qwq30b.service/embedding_api.service
```

否则：
```bash
systemctl start qwq30b.service/embedding_api.service
```

💡 注意：
- 若使用metagpt方法，需要配置API配置文件。
- 若使用react方法，需要设置API-key；使用qwen3系列需要在调用llm处设置参数'extra_body={"enable_thinking": False}'。
- 其他方法可以使用本地服务。

✅ 运行deepeval测试时，需要关闭qwq30b.service等服务，开启ollama服务并设置API-key。
```bash
systemctl stop qwq30b.service/embedding_api.service
systemctl start ollama.service
```

### 🔧 各基线方法应用
#### Metagpt方法
1. 打开配置文件config2.py，配置API-key和base_url。例如：
```python
llm:
    api_type: "dashscope"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: qwen3
    api_key: "sk-"
```

2. cd 对应目录下，将源代码解压缩到目录，并新建环境安装相关依赖包。
3. 目录下放置本次讨论需要的response.txt文件（即知识库内搜索到的问题相关内容）
4. 运行multi_qwendb2_tokentime.py脚本，输入问题，获取答案。
5. 多个问题运行multi_qwendb2_loop.py脚本，准备对应的response.txts，获取答案。

#### ReAct方法
1. cd 对应目录下，打开脚本react_tokentime_main.py，修改API-key和base_url。
2. 新建环境安装相关依赖包。
3. 目录下放置本次讨论需要的response.txt文件（即知识库内搜索到的问题相关内容）
4. 运行react_tokentime_main.py脚本，输入问题，获取答案。
5. 多个问题运行react_tokentime_loop.py脚本，准备对应的response.txts，获取答案。

#### naiverag方法
1. 新建环境安装相关依赖包，开启服务，修改对应API-key和base_url。例如：
```python
llm = ChatOpenAI(model="Qwen/Qwen3-30B-A3B",base_url="http://a6000-G5500-V6:5414/v1",api_key="EMPTY")
embedding = HuggingFaceEmbeddings(model="Alibaba-NLP/gte-multilingual-base",base_url="http://a6000-G5500-V6:5415/v1",api_key="EMPTY")
```

2. cd 对应目录下，目录下放置本次讨论需要的response.txt文件（即知识库内搜索到的问题相关内容）
3. 运行rag_tokentime_main.py脚本，输入问题，获取答案。
4. 多个问题运行rag_tokentime_loop.py脚本，准备对应的response.txts，获取答案。

#### Self-RAG方法
1. 新建环境安装相关依赖包，开启服务，修改对应API-key和base_url。
2. cd 对应目录下，目录下放置本次讨论需要的response.txt文件（即知识库内搜索到的问题相关内容）
3. 运行selfrag_tokentime_main.py脚本，输入问题，获取答案。
4. 多个问题运行selfrag_tokentime_loop.py脚本，准备对应的response.txts，获取答案。

#### LightRAG方法
1. cd 对应目录下，将源代码解压缩到目录，并新建环境安装相关依赖包。
2. 开启服务，在lightrag/llm/openai.py中修改对应API-key和base_url。
2. 目录下放置本次讨论需要的response.txt文件（即知识库内搜索到的问题相关内容）
3. 运行lightrag_openai_main.py脚本，输入问题，获取答案。建立的索引会存在dickens文件夹里。
4. 多个问题运行lightrag_openai_loop.py脚本，准备对应的response.txts，获取答案。

### 🔧 deepeval测试方法应用
1. 开启ollama服务，确认模型是否拉取。pull后执行bash命令，设置模型进行测试。
```bash
deepeval set-ollama deepseek-r1:1.5b
```

2. cd 对应目录下，将源代码解压缩到目录，并新建环境，安装deepeval依赖。
3. 设置API-key 
```bash
set OPENAI_API_KEY= sk-
```

4. 目录下放置对应conversation_data.json文件，运行test_example.py脚本（文件命名必须为test_.py），获取评测指标数值。记录日志。
```bash
deepeval test run test_example.py
```

5. 生成的result.json文件记录了测试结果。运行test_metric.py脚本，生成.xls表格文件，进行数据分析。
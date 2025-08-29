# HPC Root Cause Analysis API

该项目实现了一个用于超算系统失败作业根因计算结点定位的算法，并通过 RESTful API 对外提供服务。
算法基于跨层级因果关系图（计算结点级 + 指标级），使用随机游走（Personalized PageRank）方法，识别最可能导致作业失败的计算结点。

---

## 🔧 功能特性

- 📊 自动加载监控指标数据（CSV格式）
- 🧠 构建因果图（使用 PCMCI 与滞后相关性）
- 🔍 依据异常程度分布进行 Personalized PageRank 排名
- 🌐 提供标准 RESTful API 接口，便于集成调用
- 📦 支持可配置参数：时间窗口、关键指标、Top-K

---

## 📥 输入数据说明

算法期望的数据结构如下：

- 输入目录 `folder_path` 下应包含多个 CSV 文件，每个文件对应与失败作业相关的一个计算结点上采集的指标；
- 文件名格式建议为：`<node_id>_<metric_name>.csv`；
- 每个 CSV 文件必须包含以下两列：

|      timestamp      | metric_name |
| :-----------------: | :---------: |
| 2025-06-14 22:49:00 |    0.75    |
| 2025-06-14 22:49:15 |    0.78    |
|         ...         |     ...     |

示例（`cn61903.csv`）：

```csv

timestamp,node_arp_entries,...

2025-06-14 22:49:00,250,...

2025-06-14 22:49:15,250,...

...

```

所有指标将统一按时间对齐。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
uvicorn rca_api:app --reload --port 8000
```

---

## 📮 接口示例

### POST `/locate_root_cause`
`（例如：http://localhost:8000/locate_root_cause）`

```json
{
  "folder_path": "dataset",
  "job_id": "job_8113170",
  "failure_start": 1750129350,
  "failure_end": 1750129800,
  "failure_type": "network_bandwidth",
  "root_node": "cn61903",
  "golden_metrics": [],
  "top_k": 5
}
```

### 返回结果

```json
{
    "0": {
        "root_cause": "cn61903",
        "failure_type": "network_bandwidth",
        "top5": [
            [
                "cn61903:node_network_receive_bytes_total",
                0.28562277213130355
            ],
            [
                "cn61897:node_network_receive_bytes_total",
                0.24051390654043306
            ],
            [
                "cn61900:node_network_receive_bytes_total",
                0.23787163220132634
            ],
            [
                "cn61901:node_network_receive_bytes_total",
                0.05299224046255312
            ],
            [
                "cn61901:node_network_transmit_bytes_total",
                0.02971055174070751
            ]
        ]
    }
}
```

---

## 📂 文件结构说明

```bash
├── hpc_rca.py            # 根因分析主逻辑
├── rca_api.py            # FastAPI 接口服务定义
├── dataset/              # 示例数据文件夹
├── images/               # 结点级&指标级因果关系图
├── README.md             # 使用说明（当前文件）
├── requirements.txt      # 依赖包列表
```

---

## 📜 引用技术

- FastAPI
- NetworkX
- Tigramite（PCMCI因果发现）
- Scikit-learn
- Pandas / Numpy

---

## 📧 联系

作者：陶磊（leitao@mail.nankai.edu.cn）
单位：南开大学 软件学院

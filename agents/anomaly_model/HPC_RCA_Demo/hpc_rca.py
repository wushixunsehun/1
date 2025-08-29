def locate_root_causes(
    folder_path: str,
    job_id: str,
    failure_start: int,
    failure_end: int,
    failure_type: str,
    root_node: str,
    golden_metrics: list = None,
    top_k: int = 5
) -> dict:
    """
    Locate root cause nodes for a failed HPC job using hierarchical causal graph analysis.

    Parameters:
        folder_path: Directory path containing metric CSV files.
        job_id: Job identifier.
        failure_start: Start timestamp of the failure window (Unix).
        failure_end: End timestamp of the failure window (Unix).
        root_node: Known fault-injected node name (optional).
        golden_metrics: List of preferred metric names to analyze.
        top_k: Number of top root cause nodes to return.

    Returns:
        Dictionary containing root cause ranking results.
    """
    import os
    import pandas as pd
    import numpy as np
    import random
    from sklearn.preprocessing import MinMaxScaler
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests import parcorr
    import networkx as nx
    from itertools import combinations
    import matplotlib.pyplot as plt

    def set_random_seed(seed=2025):
        random.seed(seed)
        np.random.seed(seed)

    set_random_seed(2025)

    '''
    基于跨层级因果关系图的根因定位算法

    算法目的：通过在跨层级因果关系图上进行随机游走，定位导致作业失败的根因计算结点。

    算法步骤：
        1. 处理数据，对每个用例的每个指标：提取时间窗口数据 → 相关性筛选 → 异常分数计算 → 异常度筛选。
        2. 构建计算结点级因果图，通过网络流量指标的滞后相关性判断结点间是否连接&连接方向。
        3. 构建指标级因果图，利用PCMCI算法检测指标间因果关系。
        4. 在跨层级的因果关系图上按异常指数分布做Personalized PageRank，输出每个用例的Top 5根因候选及分数。

    输入输出：选取超算系统上的一个失败作业，将作业相关计算结点上的指标数据作为算法输入，进行根因排序
    '''

    def remove_highly_correlated(df: pd.DataFrame,
                                 golden_metrics: list,
                                 threshold: float = 0.9) -> pd.DataFrame:
        """去除高度相关的特征，只保留一个；但 golden_metrics 中的特征不被删除。"""
        metrics = [c for c in df.columns if c != 'timestamp']
        corr   = df[metrics].corr().abs()
        upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > threshold]
            for row in high_corr:
                if row in golden_metrics and col not in golden_metrics:
                    to_drop.add(col)
                elif row not in golden_metrics:
                    to_drop.add(row)

        keep = [m for m in metrics if m not in to_drop]
        return df[['timestamp'] + keep]

    def draw_causal_graph(G, title="Node Causal Graph", filename="node_causal_graph.pdf"):
        pos = nx.spring_layout(G, k=2, seed=2025)  # 用 spring 布局使图结构更清晰

        plt.figure(figsize=(10, 8))
        # 节点
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
        # 有向边
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', connectionstyle="arc3,rad=0.1")
        # 节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        # 边权重标签
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)

        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format='pdf', dpi=300)
        # print(f"图已保存为 {filename}")

        plt.show()
        plt.close()

    # 计算滞后相关性
    def lag_correlation(df1, df2, max_lag=10, method='pearson'):
        """
        计算x滞后y的滞后相关性（x领先，y滞后）

        参数：
            x, y: pd.Series，两个时间序列
            max_lag: 最大滞后步长
            method: 相关性计算方法，支持'pearson', 'spearman'

        返回：
            lag_list: 滞后步长列表（正值表示x领先y）
            corr_list: 每个滞后下的相关系数
        """
        x = df1['node_network_receive_bytes_total']
        y = df2['node_network_receive_bytes_total']
        lags = range(-max_lag, max_lag + 1)
        corr_list = []

        for lag in lags:
            if lag < 0:
                shifted_x = x.shift(-lag)
                shifted_y = y
            else:
                shifted_x = x
                shifted_y = y.shift(lag)

            df = pd.DataFrame({'x': shifted_x, 'y': shifted_y}).dropna()
            if method == 'pearson':
                corr = df['x'].corr(df['y'])
            elif method == 'spearman':
                corr = df['x'].corr(df['y'], method='spearman')
            else:
                raise ValueError("Unsupported method")
            corr_list.append(corr)

        return list(lags), corr_list

    def complete_graph_with_weight(G: nx.DiGraph, weight: float = 1.0) -> nx.DiGraph:
        """
        如果 G 只有节点、没有边，则在每对节点间添加双向边，权重相同。

        参数：
            G: networkx.DiGraph
            weight: float，要添加的边的统一权重（默认 1.0）

        返回：
            已补全的 G
        """
        if G.number_of_edges() == 0:
            nodes = list(G.nodes())
            for u, v in combinations(nodes, 2):
                # 添加 u->v 和 v->u
                G.add_edge(u, v, weight=weight)
                # G.add_edge(v, u, weight=weight)
        return G

    def build_dag_causal_graph(case_data, max_lag=30):
        G = nx.DiGraph()

        nodes = list(case_data.keys())

        for node in nodes:
            G.add_node(node)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):  # 确保只遍历一次
                node_i = nodes[i]
                node_j = nodes[j]
                series_i = case_data[node_i]
                series_j = case_data[node_j]

                lags, corrs = lag_correlation(series_i, series_j, max_lag)
                max_corr = max(corrs, key=abs)
                best_lag = lags[corrs.index(max_corr)]
                if best_lag > 0:
                    G.add_edge(node_i, node_j, weight=abs(max_corr))
                elif best_lag < 0:
                    G.add_edge(node_j, node_i, weight=abs(max_corr))

        return G
    
    def draw_hierarchical_graph(G: nx.DiGraph, filename: str):
        """
        绘制带有 intra/inter 边的分层因果图并保存为 PDF
        G: 包含节点之间 'type' 属性 = 'intra' 或 'inter' 的有向图
        filename: 输出 PDF 文件名
        """
        # 布局：圆形布局
        pos = nx.circular_layout(G)

        plt.figure(figsize=(16, 12))

        # 根据边的 'type' 属性区分 intra/inter 边
        intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'intra']
        inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'inter']
        inter_self_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'inter_self']

        # 节点
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=8)

        # intra-node 边：绿色实线
        nx.draw_networkx_edges(
            G, pos,
            edgelist=intra_edges,
            edge_color='green',
            arrows=True,
            arrowstyle='->',
            arrowsize=20,
            width=2
        )
        # inter-node 边：红色虚线
        nx.draw_networkx_edges(
            G, pos,
            edgelist=inter_edges,
            edge_color='red',
            style='dashed',
            arrows=True,
            arrowstyle='->',
            arrowsize=20,
            width=2
        )

        # inter-node 自环：蓝色点线
        nx.draw_networkx_edges(
            G, pos,
            edgelist=inter_self_edges,
            edge_color='blue',
            connectionstyle='arc3,rad=0.4',
            arrowsize=10,
            width=2
        )

        # 边标签：显示 weight
        edge_labels = {
            (u, v): f"w={d.get('weight', 0):.4f}"
            for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='gray',
            font_size=7,
        )

        plt.title("Hierarchical Causal Graph (Metric Level)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format='pdf', dpi=300)
        plt.show()
        plt.close()
        # print(f"图已保存为 {filename}")

    def process_metric_graph(metric_graph: nx.DiGraph) -> nx.DiGraph:
        """
        1) 去除孤立节点
        2) 保留最大连通子图（基于无向连通性）
        3) 反向所有边，保留原有属性（如 weight）
        """
        # 1. 去除孤立节点
        g = metric_graph.copy()
        isolates = list(nx.isolates(g))
        if isolates:
            g.remove_nodes_from(isolates)

        # 2. 找出最大连通子图（先转为无向图）
        undirected = g.to_undirected()
        components = list(nx.connected_components(undirected))
        if not components:
            raise ValueError("图中没有任何连通组件")
        largest_cc = max(components, key=len)
        g_lcc = g.subgraph(largest_cc).copy()

        # 3. 反向所有边（保留属性）
        g_rev = g_lcc.reverse(copy=True)

        return g_rev

    def rank_by_weighted_betweenness(G):
        # 1. 为每条边定义“length”属性 = 1/weight
        for u, v, d in G.edges(data=True):
            w = d.get('weight', 0.0) or 1e-6  # 防止零权重
            d['length'] = 1.0 / w

        # 2. 计算加权中介中心性
        bc = nx.betweenness_centrality(
            G,
            weight='length',   # 使用 length 作为路径长度
            normalized=True    # 归一化
        )
        # 3. 返回按 centrality 降序
        return sorted(bc.items(), key=lambda x: x[1], reverse=True)

    def rank_by_personalized_pagerank(G, entry_node, alpha=0.85, max_iter=1000, tol=1e-06):
        # 1. personalization 只在入口节点上设置 1
        personalization = {n: 0.0 for n in G.nodes()}
        personalization[entry_node] = 1.0

        # 2. 运行 PageRank（带权重）
        pr = nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            weight='weight',
            max_iter=max_iter,
            tol=tol
        )
        # 3. 返回按 score 降序排序的列表
        return sorted(pr.items(), key=lambda x: x[1], reverse=True)

    def rank_by_anomaly_distribution(G: nx.DiGraph,
                                     scores_df: pd.DataFrame,
                                     alpha: float = 0.85,
                                     max_iter: int = 1000,
                                     tol: float = 1e-06) -> dict:
        """
        用所有异常分数构建 personalization 分布来跑 PageRank。

        scores_df: DataFrame with columns ['node','metric','anomaly_score']，
                   node: e.g. 'cn61897', metric: e.g. 'node_load1'
        """
        # 1) 构造 personalization 分布
        #    只考虑在图 G 中实际存在的节点（格式 node:metric）
        merged = []
        for _, row in scores_df.iterrows():
            nm = f"{row['node']}:{row['metric']}"
            if nm in G:
                merged.append((nm, row['anomaly_score']))
        if not merged:
            raise RuntimeError("没有任何异常指标在图中匹配到")

        # 归一化为概率
        total = sum(sc for _, sc in merged)
        personalization = { n: 0.0 for n in G.nodes() }
        for n, sc in merged:
            personalization[n] = sc / total

        # 2) 跑带权重的 Personalized PageRank
        pr = nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            weight='weight',
            max_iter=max_iter,
            tol=tol
        )

        return sorted(pr.items(), key=lambda x: x[1], reverse=True)


    # 用于存储所有读取到的 DataFrame，key 为文件名，value 为 DataFrame
    dataframes = {}
    # 遍历目录及其子目录
    scaler = MinMaxScaler()
    # 输入指标数据所在文件夹路径
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"指定的文件夹路径不存在：{folder_path}")
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    # 读取 CSV
                    df = pd.read_csv(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['timestamp'] = df['timestamp'].astype(int) // 10**9
                    df = df.dropna(subset=['timestamp'])  # 删除不能转换的行
                    # 归一化数据
                    columns_to_normalize = df.columns[df.columns != 'timestamp']
                    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
                    dataframes[filename] = df
                    # print(f"已加载: {filename}，形状: {df.shape}")
                except Exception as e:
                    print(f"读取 {filename} 时出错：{e}")

    cases = [
            {
            "job_id": job_id,
            "start": failure_start,
            "end": failure_end,
            "failure_type": failure_type,
            "root_node": root_node
        }
    ]

    if golden_metrics is None or len(golden_metrics) == 0:
        # 默认指标列表
        golden_metrics = ['node_load1', 'node_cpu_seconds_total', 'node_memory_MemFree_bytes', 'node_memory_AnonPages_bytes', 'node_network_transmit_bytes_total', 'node_network_receive_bytes_total', 'node_filesystem_avail_bytes']

    # 1 处理指标数据，筛选符合要求的指标

    # 参数配置
    threshold_corr  = 0.9   # 去相关阈值
    threshold_score = 0.05  # 异常分数阈值
    extracted_cases = []

    for i, case in enumerate(cases):
        # 1) 计算时间窗口
        duration = case['end'] - case['start']
        start_ts = case['start'] - 3 * duration
        end_ts   = case['end']   + 2 * duration

        # 2) 对每个节点：提窗口 → 相关性筛选 → 异常分数计算 → 异常度筛选
        metrics_by_node = {}
        scores = []
        for fname, df in dataframes.items():
            node = fname.split('.')[0]
            if not set(golden_metrics).issubset(df.columns):
                print(f"[跳过] 节点 {node} 缺少 golden_metrics 中的部分指标")
                continue

            window_df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].copy()
            filtered  = remove_highly_correlated(window_df, golden_metrics, threshold_corr)

            # 计算每个指标的 3-sigma 异常分数
            metrics = [c for c in filtered.columns if c != 'timestamp']
            μ = filtered[metrics].mean()
            σ = filtered[metrics].std()
            mask   = (filtered[metrics] - μ).abs() > 3 * σ
            counts = mask.sum()
            keep_metrics = []
            for m in metrics:
                sc = counts[m] / len(filtered)
                if sc >= threshold_score or m in golden_metrics:
                    keep_metrics.append(m)
                    scores.append({'node': node, 'metric': m, 'anomaly_score': sc})
            # 最终存入 metrics_by_node：只保留筛选后的指标
            # print(filtered[['timestamp'] + keep_metrics].shape, f"Node {node} → {len(keep_metrics)} metrics after filtering")
            metrics_by_node[node] = filtered[['timestamp'] + keep_metrics]

        # 如果没有任何符合阈值的指标，跳过此 case
        if not scores:
            continue

        # 3) 汇总分数并选最异常
        scores_df = pd.DataFrame(scores)
        scores_df.sort_values('anomaly_score', ascending=False, inplace=True)
        top = scores_df.iloc[0]
        top_node   = top['node']
        top_metric = top['metric']
        top_score  = top['anomaly_score']

        # 4) 在该节点的最终 window 中，选关联度最高的 golden_metrics
        df_top = metrics_by_node[top_node]
        common = [g for g in golden_metrics if g in df_top.columns]
        corrs = {g: abs(df_top[[top_metric, g]].corr().iloc[0,1]) for g in common}
        entry_metric = max(corrs, key=corrs.get)
        entry_corr   = corrs[entry_metric]

        # 5) 聚合到 extracted_cases
        extracted_cases.append({
            'job_id'          : case.get('job_id', 'unknown'),
            'case_id'         : i,
            'root_node'       : case['root_node'],
            'failure_type'    : case['failure_type'],
            'start_ts'        : start_ts,
            'end_ts'          : end_ts,
            'metrics_by_node' : metrics_by_node,   # 最终保留：相关性+异常度筛后的 DataFrames
            'scores_df'       : scores_df,         # 仅含最终 metrics_by_node 的分数
            'top_node'        : top_node,
            'top_metric'      : top_metric,
            'top_score'       : top_score,
            'entry_metric'    : entry_metric,
            'entry_corr'      : entry_corr
        })

    # 查看示例
    # for ec in extracted_cases:
    #     print(f"Job {ec['job_id']}, Case {ec['case_id']}：")
    #     print(f" Window {ec['start_ts']}–{ec['end_ts']}")
    #     print(f" Top anomaly → Node {ec['top_node']}, Metric {ec['top_metric']} (score={ec['top_score']:.3f})")
    #     print(f" Entry point → {ec['entry_metric']} (corr={ec['entry_corr']:.3f})")


    #2 构建结点级因果图
    # 连接关系，通过网络流量指标判断结点间是否连接&连接方向

    node_graphs = {}
    # 遍历每个故障 case，构建因果关系图
    for case in extracted_cases:
        # print(f"构建故障 case {case['case_id']} 的因果关系图：")

        G = build_dag_causal_graph(case['metrics_by_node'], max_lag=10)  # 构建有向无环图
        G = complete_graph_with_weight(G, weight=0.5)  # 确保图是完全的
        node_graphs[case['case_id']] = {'graph': G}


        # 打印因果关系图的边（节点间的因果关系）
        # print("因果关系图的边（边的方向表示因果关系）：")
        # print(G.edges())
        # draw_causal_graph(G, title=f"Causal Graph for {case['case_id']}", filename=f"images/{case['job_id']}_{case['case_id']}_node_causal_graph.pdf")

    #3 构建指标级因果图
    # PCMCI算法检测指标间因果关系，考虑inter-node & intra-node

    # 参数：PCMCI 设置
    max_lag = 5               # 最大滞后步长
    alpha_level = 0.1        # 显著性水平

    for case in extracted_cases:
        case_id = case['case_id']
        job_id = case['job_id']
        intra_graphs = {}  # 暂存每个节点的指标级图
        scores_df = case['scores_df']

        # 1) 对每个节点，运行 PCMCI
        for node, df in case['metrics_by_node'].items():
            # df 包含 timestamp + 若干指标列
            metrics = [c for c in df.columns if c != 'timestamp']
            if len(metrics) < 2:
                # 少于两个指标，无法做因果检测
                intra_graphs[node] = nx.DiGraph()
                continue

            # 准备数据：去掉 timestamp，以 numpy array 形式喂给 PCMCI
            data_arr = df[metrics].values
            dataframe = pp.DataFrame(data_arr, datatime=df['timestamp'].values)

            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=parcorr.ParCorr(significance='analytic'),
                verbosity=0
            )
            results = pcmci.run_pcmci(
                tau_max=max_lag,
                pc_alpha=alpha_level  # 使用后续 p-value 检验
            )

            # 根据显著性水平筛选因果链接
            q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
            pcmci_links = {}
            # results['val_matrix'][tgt, src, lag]
            for tgt_idx, src_idx, lag in zip(*np.where(q_matrix <= alpha_level)):
                if src_idx == tgt_idx:
                    continue
                corr_val = results['val_matrix'][tgt_idx, src_idx, lag]
                src_metric = metrics[src_idx]
                tgt_metric = metrics[tgt_idx]

                # 只保留正向或负向均可
                if lag > 0:
                    # src_metric (t - lag) → tgt_metric (t)
                    pcmci_links.setdefault((src_metric, tgt_metric), []).append((lag, corr_val))
                elif lag < 0:
                    # tgt leads src, 反向存
                    pcmci_links.setdefault((tgt_metric, src_metric), []).append((-lag, corr_val))

            # 构建该节点的指标级因果图
            Gm = nx.DiGraph()
            Gm.add_nodes_from(metrics)
            for (src_metric, tgt_metric), edges in pcmci_links.items():
                # 如果同一对有多个 lag/link，取相关度最大的那条
                best = max(edges, key=lambda x: abs(x[1]))
                Gm.add_edge(src_metric, tgt_metric,
                           weight=abs(best[1]),
                           lag=best[0])
            intra_graphs[node] = Gm

        # 2) 合并所有节点的 intra-node 图，并加入 inter-node 边
        metric_graph = nx.DiGraph()
        # 首先把所有指标级节点加进去
        for node, Gm in intra_graphs.items():
            for metric in Gm.nodes:
                metric_graph.add_node(f"{node}:{metric}")

        # 添加 intra-node 边
        for node, Gm in intra_graphs.items():
            for src, tgt, data in Gm.edges(data=True):
                metric_graph.add_edge(f"{node}:{src}",
                                      f"{node}:{tgt}",
                                      **data)

        # 添加 inter-node 边——基于 node_network_receive_bytes_total 的节点图
        node_G = node_graphs[case_id]['graph']
        for src_node, tgt_node, data in node_G.edges(data=True):
            # 连接对应的度量：“node_network_receive_bytes_total”
            src_metric = f"{src_node}:node_network_receive_bytes_total"
            tgt_metric = f"{tgt_node}:node_network_receive_bytes_total"
            # 这里复用原来节点图的权重作为边的 weight
            metric_graph.add_edge(src_metric, tgt_metric,
                                  weight=data.get('weight', None),
                                  inter_node=True)

        for u, v, d in metric_graph.edges(data=True):
            d['type'] = 'inter' if d.get('inter_node') else 'intra'

        processed_graph = process_metric_graph(metric_graph)

        # 先找出所有参与 inter-node 边传播的节点
        inter_nodes = set()
        for u, v, d in processed_graph.edges(data=True):
            if d.get('type') == 'inter':
                inter_nodes.add(u)
                inter_nodes.add(v)
        # 给这些节点添加自环，weight 从 scores_df 里取 anomaly_score
        for nm in inter_nodes:
            node, metric = nm.split(':', 1)
            # 找 anomaly_score
            row = scores_df[(scores_df['node'] == node) & (scores_df['metric'] == metric)]
            if not row.empty:
                sc = row['anomaly_score'].iloc[0]
                # 添加（或更新）自环
                processed_graph.add_edge(nm, nm, weight=sc, type='inter_self')
            # else:
            #     # 若未找到分数，可选择跳过或设为 0
            #     processed_graph.add_edge(nm, nm, weight=0.0, type='inter_self')

        # 存回 node_graphs
        node_graphs[case_id]['metric_graph'] = processed_graph

        # print(f"[Case {case_id}] 构建完毕指标级因果图，共 {processed_graph.number_of_nodes()} 个节点，{processed_graph.number_of_edges()} 条边。")

        # 然后调用绘图函数
        # draw_hierarchical_graph(
        #     processed_graph,
        #     filename=f"images/{job_id}_{case_id}_metric_graph.pdf"
        # )

    #4 跨层级随机游走定位根因

    # 参数：PageRank 的阻尼系数
    alpha = 0.85
    root_causes = {}

    for case in extracted_cases:
        cid       = case['case_id']
        G         = node_graphs[cid]['metric_graph']
        top_node  = case['top_node']
        entry_met = case['entry_metric']
        scores_df = case['scores_df']
        entry_node = f"{top_node}:{entry_met}"
        # # 如果不存在该指标，就降级到 node_network_receive_bytes_total
        # if entry_node not in G:
        #     entry_node = f"{top_node}:node_network_receive_bytes_total"
        # print(f"Case {cid}：入口节点 = {entry_node}")
        # pagerank_ranking = rank_by_personalized_pagerank(G, entry_node, alpha)
        # pagerank_ranking = rank_by_weighted_betweenness(G)
        pagerank_ranking = rank_by_anomaly_distribution(G, scores_df, alpha=alpha, max_iter=5000)

        # 排序并取 Top 5
        if len(pagerank_ranking) <= top_k:
            top_k = len(pagerank_ranking)
        results = pagerank_ranking[:top_k]
        root_causes[cid] = {
            'root_cause': case['root_node'],
            'failure_type': case['failure_type'],
            'top5': results
        }

    # for cid, info in root_causes.items():
    #     print(f"Case {cid}：根因节点 = {info['root_cause']}")
    #     print(f"故障类型 = {info['failure_type']}\n")
    #     print(f"Top {top_k} 根因候选及分数：")
    #     for rank, (node, score) in enumerate(info['top5'], 1):
    #         print(f"  {rank}. {node} (score = {score:.4f})")
    #     print()

    return root_causes
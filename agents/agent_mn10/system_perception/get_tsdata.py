#!/usr/bin/python3
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from datetime import datetime, timedelta
import random

# Prometheus 服务器 URL
prometheus_url = "http://ion211:10900"
initial_step = "15s"  # 初始步长为 15 秒
max_connections = 50
initial_connections = 10
max_retries = 5
retry_delay = 5  # 秒

# 生成节点列表
nodes = [f"oss{i}" for i in range(20, 25)]

metrics = [   ]
# 固定的时间范围
start_time = "2025-01-14T16:00:00Z"
end_time = "2025-01-15T16:00:00Z"

# 获取时间序列数据的函数
def get_time_series_data(prometheus_url, metric_name, host, cluster, start_time, end_time, step):
    query = f'{metric_name}{{host="{host}", cluster="{cluster}"}}'
    query_url = f"{prometheus_url}/api/v1/query_range?query={query}&start={start_time}&end={end_time}&step={step}"
    for attempt in range(max_retries):
        try:
            response = requests.get(query_url)
            # 记录请求 URL 和响应状态
            log_request(query_url, response)
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 400 and 'exceeded maximum resolution' in response.text:
                print(f"{metric_name} 的步长 {step} 超过最大分辨率。正在减少步长。")
                new_step = reduce_step_size(step)
                return get_time_series_data(prometheus_url, metric_name, host, cluster, start_time, end_time, new_step)
            else:
                print(f"获取 {host} 上 {metric_name} 的时间序列数据失败（尝试 {attempt + 1}/{max_retries}）：HTTP {response.status_code}")
                print(f"响应内容：{response.text}")
        except requests.exceptions.RequestException as e:
            print(f"获取 {host} 上 {metric_name} 的时间序列数据失败（尝试 {attempt + 1}/{max_retries}）：{e}")
        
        # 使用指数退避算法
        backoff_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
        time.sleep(backoff_time)
    return None

# 记录请求 URL 和响应状态的函数
def log_request(query_url, response):
    log_message = f"URL: {query_url}, Status Code: {response.status_code}, Response: {response.text[:200]}..."  # 记录响应的前 200 个字符
    with open('data01_02.txt', 'a') as log_file:
        log_file.write(log_message + '\n')
    print(log_message)

# 动态减少步长的函数
def reduce_step_size(step):
    if 's' in step:
        seconds = int(step.replace('s', ''))
        if seconds > 1:
            return f"{seconds // 2}s"
    elif 'm' in step:
        minutes = int(step.replace('m', ''))
        if minutes > 1:
            return f"{minutes // 2}m"
        else:
            return f"{minutes * 60}s"
    elif 'h' in step:
        hours = int(step.replace('h', ''))
        return f"{hours * 30}m"
    return step

# 将数据保存到本地文件的函数
def save_data_to_file(data, directory, filename):
    try:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"数据已写入 {filepath}")
    except IOError as e:
        print(f"写入文件失败：{e}")

# 使用并发获取数据的函数
def fetch_data_with_concurrency(prometheus_url, metrics, node, start_time, end_time, step, max_connections,
                                initial_connections, save_directory):
    current_connections = initial_connections

    for metric in metrics:
        success = False
        while current_connections <= max_connections and not success:
            try:
                with ThreadPoolExecutor(max_workers=current_connections) as executor:
                    future = executor.submit(get_time_series_data, prometheus_url, metric, node, "TH-eX", start_time, end_time, step)
                    data = future.result()
                    if data:
                        # 为每个节点创建目录
                        node_directory = os.path.join(save_directory, node)
                        if not os.path.exists(node_directory):
                            os.makedirs(node_directory)
                        filename = f'{metric.replace("/", "_")}.json'
                        save_data_to_file(data, node_directory, filename)
                        print(f"{node} 上 {metric} 的数据已使用 {current_connections} 个连接成功保存。")
                        success = True
            except KeyboardInterrupt:
                print("用户中断进程。")
                break
            except Exception as e:
                print(f"{node} 使用 {current_connections} 个连接时发生异常：{e}")
                current_connections = max(1, current_connections // 2)

# 主函数
def main():
    save_directory = "/thfs1/home/sxs/data_12_16/data01_15_1/TH-eX"  # 设置基础数据保存目录
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for node in nodes:
        fetch_data_with_concurrency(prometheus_url, metrics, node, start_time, end_time, initial_step, max_connections,
                                    initial_connections, save_directory)

if __name__ == "__main__":
    main()


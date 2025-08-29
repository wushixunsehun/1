import requests

proxies = {
    "http": "",
    "https": "",
}

url = "http://localhost:5010/query"
data = {
    "query": "TH3F系统如何安装和使用可视化工具NCL？",
    "source": '',
    "topk": 4
}

try:
    response = requests.post(url, json=data, proxies=proxies)
    response.raise_for_status()
    print("✅ Success!")
    print(response.json())
except requests.exceptions.HTTPError as e:
    print(f"❌ Request failed with status code: {response.status_code}")
    print(response.text)


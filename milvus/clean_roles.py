import os
import sys
from typing import List
from pymilvus import MilvusClient, MilvusException


# ---------- 基础配置 ----------
URI = 'http://192.168.10.138:19530'
TOKEN = 'root:lhltxh971012'

def main():

    client = MilvusClient(uri=URI, token=TOKEN)
    meta = client.backup_rbac()  # 一行生成 JSON
    with open("rbac_backup.json","w") as f:
        f.write(meta)

    with open("rbac_backup.json") as f:
        client.restore_rbac(data=f.read())

if __name__ == "__main__":
    main()
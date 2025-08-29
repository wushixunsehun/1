#!/bin/bash

# 定义远程服务器和本地路径
REMOTE_HOST="a6000"
REMOTE_PATHS=("/home/tanxh/mas/agents/agent_mn10/system_perception/system_main_stable.py" "/home/tanxh/mas/agents/agent_mn10/system_perception/experts/state_expert_stable.py" "/home/tanxh/mas/agents/agent_mn10/strategy_plan/strategy_main_stable.py" "/home/tanxh/mas/agents/config.yaml")
LOCAL_PATHS=("/root/agents/agent_mn10/system_perception/system_main_stable.py" "/root/agents/agent_mn10/system_perception/experts/state_expert_stable.py" "/root/agents/agent_mn10/strategy_plan/strategy_main_stable.py" "/root/agents/config.yaml")

# 定义需要重启的 systemd 服务名
SERVICES=("masAPI-strat_plan.service" "masAPI-sys_perc.service")

# 拷贝文件
for i in {0..3}; do
    echo "拷贝 ${REMOTE_HOST}:${REMOTE_PATHS[$i]} 到本机 ${LOCAL_PATHS[$i]}"
    scp "${REMOTE_HOST}:${REMOTE_PATHS[$i]}" "${LOCAL_PATHS[$i]}"
    if [ $? -ne 0 ]; then
        echo "拷贝失败: ${REMOTE_HOST}:${REMOTE_PATHS[$i]}"
        exit 1
    fi
done

# 重启 systemd 服务
for svc in "${SERVICES[@]}"; do
    echo "重启服务: $svc"
    sudo systemctl restart "$svc"
    sleep 2
done

sleep 5
# 检查日志
for svc in "${SERVICES[@]}"; do
    echo "检查 $svc 日志:"
    last_line=$(sudo journalctl -u "$svc" -n 1 --no-pager)
    echo "$last_line"
    if echo "$last_line" | grep -q "Uvicorn running on"; then
        echo "$svc 重启成功"
    else
        echo "$svc 重启失败"
    fi
done
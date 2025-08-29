#!/bin/bash

# ========= 用户参数 =========
MODULE_NAME="command_main_stable"  # 主程序名（不含 .py）
ENTRY_FILE="agent_mn10/command_run/${MODULE_NAME}.py"  # 入口文件路径
DIST_DIR="./dist/${MODULE_NAME}"  # 输出路径
BUILD_DIR="./build/${MODULE_NAME}"  # 构建路径
SPEC_PATH="${BUILD_DIR}"  # .spec 文件输出路径
OUTPUT_NAME="agent_run"  # 最终可执行文件名称

# ========= 自动清理旧文件 =========
echo "🚮 正在清理旧的构建文件..."
rm -rf "${DIST_DIR}" "${BUILD_DIR}" "${MODULE_NAME}.spec"

# ========= 执行打包 =========
echo "🚀 正在打包模块：${MODULE_NAME} ..."
pyinstaller --onefile --clean --strip \
  --name "${OUTPUT_NAME}" \
  --distpath "${DIST_DIR}" \
  --workpath "${BUILD_DIR}" \
  --specpath "${SPEC_PATH}" \
  "${ENTRY_FILE}"

# ========= 打包结果提示 =========
if [[ -f "${DIST_DIR}/${OUTPUT_NAME}" ]]; then
  echo "✅ 打包成功：${DIST_DIR}/${OUTPUT_NAME}"
else
  echo "❌ 打包失败，请检查路径或导入错误。"
fi

#!/bin/bash

# 1. 自动添加所有文件 (包括 LFS 大文件)
echo "📦 正在添加文件..."
git add .

# 2. 自动提交
# 如果你运行脚本时没写备注，自动用当前时间做备注
commit_msg="$1"
if [ -z "$commit_msg" ]; then
    commit_msg="Auto update: $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "📝 正在提交... 备注: $commit_msg"
git commit -m "$commit_msg"

# 3. 推送
echo "🚀 正在推送到远程服务器..."
git push

echo "✅ 完成！"



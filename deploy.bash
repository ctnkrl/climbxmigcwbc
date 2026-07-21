# 下载最新分支
git clone --branch main https://git.x-humanoid-cloud.com/motion-intelligence-group/xmigcs.git .

# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装uv环境xmigcs
uv sync --no-install-project
export XMIGCS_DEV=1
uv pip install -e .

# 安装evt2串并联脚踝包
wget http://10.10.250.160:5000/download/evt2%E8%84%9A%E8%B8%9D/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl
uv pip install sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl

# 准备启动
# 检查确保没有其他body节点在运行
ps -aux | grep body
bash startup_nav_real.sh

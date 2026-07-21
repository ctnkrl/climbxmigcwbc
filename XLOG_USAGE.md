### 1. 解压二进制包

```bash
# 创建文件夹
mkdir -p ~/xlogs_depends/xlog_v1.0.1
# 解压到路径下
# humble
tar -C ~/xlogs_depends/xlog_v1.0.1 -xf downloads/xlog_v1.0.1_standalone_humble_linux_x86_64.tar.gz
# jazzy
tar -C ~/xlogs_depends/xlog_v1.0.1 -xf downloads/xlog_v1.0.1_standalone_jazzy_linux_x86_64.tar.gz
```


### 2. 设置环境变量

```bash
#每次打开终端均需要设置环境变量，可写入bashrc或zshrc
# NOTE: 注意用户名eai改成对应的
cat <<'EOF' >> ~/.bashrc
export XLOG_PREFIX=/home/eai/xlogs_depends/xlog_v1.0.1
export CMAKE_PREFIX_PATH=$XLOG_PREFIX:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$XLOG_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$XLOG_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
EOF

cat <<'EOF' >> ~/.zshrc
export XLOG_PREFIX=/home/eai/xlogs_depends/xlog_v1.0.1
export CMAKE_PREFIX_PATH=$XLOG_PREFIX:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$XLOG_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$XLOG_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
EOF

source ~/.zshrc
source ~/.bashrc
```

### 3. xmigcs运行

状态机已接入xlogs， 运行xmigs第一行出现下面输出说明xlog配置完成。

```bash
[XLOG][FSM] initialized successfully: logger_id=xmigcs_fsm, sub_dir=xmigcs
```

记录的状态机log信息将在ctrl c 后落盘到/home/eai/logs/glogs/xmigcs目录下。



# Deployment GMR Routes Handoff

生成时间：2026-07-21

## 背景

已阅读 Codex 线程：

- 线程名：继续 LAFAN locomotion 分支
- 线程 id：019f3aad-235d-7111-aeb1-61228b81c35b
- 会话文件：/home/mig/.codex/sessions/2026/07/07/rollout-2026-07-07T11-44-13-019f3aad-235d-7111-aeb1-61228b81c35b.jsonl

该分支主要处理 Holosoma/LAFAN locomotion 的 reference 生成和 BeyondMimic 训练准备：

- 35 条泛化动作缩到 9 条候选动作。
- 003/008/005 分别对应不同箱高组。
- 005 当前相对可用；003 有左右晃；008 重点是起始支撑和接触相位。
- 训练侧要求等 9 条 NPZ 视觉审核通过后再放进 BeyondMimic。

该分支没有看到已经完成 xmigcs-dev 的 GMR 部署双路线接入。

## 当前部署路线

这次改动把 stairs 策略的 reference 来源明确拆成两条路线：

- `embedded_npz`：路线 A，默认路线。ONNX 内嵌多 NPZ reference，通过 `motion_id + local_step` 选择动作帧。
- `realtime_gmr`：路线 B，实时路线。从 `RobotData` 的 realtime reference cache 读取在线 reference。

当前默认仍是：

```yaml
reference_source: embedded_npz
use_realtime_gmr: false
```

所以当前实机单 motion / 多 motion 离线播放不会被改变。

## 本目录包含的部署代码

文件保持和 xmigcs-dev 原路径一致：

```text
src/xmigcs/common/robot_data.py
src/xmigcs/policy/stairs/fsm_stairs.py
src/xmigcs/policy/stairs/config/stairs.yaml
```

对应当前工作区文件：

```text
/home/mig/xmigcs-dev/src/xmigcs/common/robot_data.py
/home/mig/xmigcs-dev/src/xmigcs/policy/stairs/fsm_stairs.py
/home/mig/xmigcs-dev/src/xmigcs/policy/stairs/config/stairs.yaml
```

## 已完成

- `stairs.yaml` 增加 `reference_source` 和 `gmr_reference.max_age_s`。
- `fsm_stairs.py` 读取 `reference_source`，支持 `embedded_npz` / `realtime_gmr`。
- `fsm_stairs.py` 进入状态时打印 `reference_source`，便于确认当前走哪条路线。
- `fsm_stairs.py` 的 realtime 分支优先读取 `RobotData.get_gmr_reference(max_age_s=...)`。
- `robot_data.py` 提供线程安全 realtime reference cache：
  - `configure_gmr_reference`
  - `set_gmr_reference`
  - `get_gmr_reference`
- 已通过 `python3 -m py_compile` 语法检查。

## 还没完成

- 还没有在 `robot_interface.py` 接 `/gmr_info` subscriber。
- 还没有确认 `/gmr_info` 的真实 ROS 消息类型、joint/body 顺序、坐标系和频率。
- 还没有把 `stairs.yaml` 切到 `reference_source: realtime_gmr`。
- 没有改 ONNX、训练代码、蒸馏代码、embedded NPZ 播放逻辑。

## 后续接 realtime GMR 的最小路径

1. 在实际上机环境确认 `/gmr_info`：

```bash
ros2 topic info /gmr_info -v
ros2 interface show <消息类型>
```

2. 只在确认消息类型后，给 `/home/mig/xmigcs-dev/src/xmigcs/common/robot_interface.py` 加最小 subscriber。

3. subscriber 解析出和训练 reference 同 schema 的数据，并写入：

```python
robot_data.set_gmr_reference(
    joint_pos=...,
    joint_vel=...,
    body_pos_w=...,
    body_quat_w=...,
    body_lin_vel_w=...,
    body_ang_vel_w=...,
)
```

4. shadow 检查 reference 维度、顺序、频率和 stale 计数。

5. 确认后再切：

```yaml
reference_source: realtime_gmr
use_realtime_gmr: true
```

## 注意

不要把 `embedded_npz` 路线说成实时 GMR。当前已经可用的是路线 A；路线 B 只有 FSM/cache 入口，真实 `/gmr_info` 数据源还没接。

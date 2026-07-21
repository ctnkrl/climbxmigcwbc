# climbxmigcwbc

XMIGC deployment handoff for the climb WBC / BeyondMimic policy route split.

## What is included

This repository contains the current `/home/mig/xmigcs-dev` deployment-side changes for the stairs policy:

- `src/xmigcs/common/robot_data.py`
- `src/xmigcs/policy/stairs/fsm_stairs.py`
- `src/xmigcs/policy/stairs/config/stairs.yaml`
- `docs/deployment_gmr_routes_20260721.md`

## Route A: embedded NPZ

Default route. The ONNX stores multi-motion NPZ references, and `stairs.yaml` selects frames with `motion_id + local_step`.

Current default:

```yaml
reference_source: embedded_npz
use_realtime_gmr: false
```

This route is preserved for deterministic 0.2m / 0.4m / 0.6m real-machine reproduction of jitter, sliding, and foot-hooking issues.

## Route B: realtime GMR

The stairs FSM can now read realtime reference frames from `RobotData.get_gmr_reference(...)` when:

```yaml
reference_source: realtime_gmr
use_realtime_gmr: true
```

This is only the deployment-side cache/FSM entry point. The real `/gmr_info` subscriber is not wired yet, because the true message type, field layout, joint/body order, coordinate frame, and timing still need to be confirmed in the actual xmigcs-dev runtime.

## Verification

The copied deployment files pass:

```bash
python3 -m py_compile src/xmigcs/common/robot_data.py src/xmigcs/policy/stairs/fsm_stairs.py
```

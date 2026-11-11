# LIFT3

Mobile manipulator: omni-wheel chassis, lift platform, two 6-DoF arms with parallel grippers.
This repository holds the robot description, a web viewer, and reinforcement-learning code that
trains a whole-body policy to bring both grippers onto two target frames with low arm torque.

## Layout

| Path | Contents |
|---|---|
| `LIFT3/LIFT3/` | ROS package: `urdf/LIFT3.urdf`, `meshes/`, launch files |
| `viewer/` | Web viewer (three.js, no build step): joints, targets, policy replay, live policy |
| `rl/` | MuJoCo model builder, GPU training environment, PPO trainer, evaluation, live server |
| `rl/CONTROL_INTERFACE.md` | What the policy outputs and how it becomes motor commands |
| `angle_limit.jpg` | Joint limit / torque / speed reference for the arm |

## Viewer only

```bash
python3 -m http.server 43029        # run in the repository root
# open http://127.0.0.1:43029/viewer/
```

## Training (GPU)

Needs a Python environment with `mujoco`, `mujoco_warp`, `warp-lang` and `torch` (CUDA).

```bash
cd rl
python build_model.py                       # URDF -> lift3.xml / lift3_vec.xml
python train_gpu.py --name my_run --iters 3000
python baseline.py --n 200                  # optimiser baseline targets (CPU)
python evaluate.py --ckpt runs/my_run/model_last.pt
```

## Live policy in the viewer

```bash
cd rl
python live_server.py --ckpt runs/v13/model_final.pt --port 43030
# open http://127.0.0.1:43030/viewer/ and press "Start live policy"
# add --host 0.0.0.0 to reach it from other devices on the network (no password: trusted networks only)
```

## Status

Work in progress. Arm motor torque limits follow `angle_limit.jpg`; PD gains, the lift force limit
and the wheel model are simulation assumptions (see `rl/CONTROL_INTERFACE.md`) and have not been
checked on the real robot.

"""Evaluate a trained policy on the baseline target set and export rollouts for the web viewer.

    python3 evaluate.py --ckpt runs/v1/model_last.pt            -> prints metrics, writes rollouts.json
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

from lift3_env import DT_CTRL, EPISODE_STEPS, HERE, POS_TOL, ROT_TOL
from lift3_gpu import Lift3GPU
from train_gpu import ActorCritic


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).ravel())
    return [q[1], q[2], q[3], q[0]]          # x y z w (three.js order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "v9", "model_last.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--export", type=int, default=6, help="number of rollouts written to rollouts.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, weights_only=False, map_location="cuda:0")
    ac = ActorCritic().to("cuda:0")
    ac.load_state_dict(ck["model"])
    ac.eval()
    base = np.load(os.path.join(HERE, "baseline_targets.npz"))
    n = min(args.n, len(base["tgt_pos"]))
    env = Lift3GPU(n, seed=777)          # all target pairs are evaluated in parallel
    m = env.mjm
    names = [m.joint(j).name for j in range(m.njnt)]
    env.tgt_pos[:] = torch.tensor(base["tgt_pos"][:n], dtype=torch.float32, device="cuda:0")
    env.tgt_R[:] = torch.tensor(base["tgt_R"][:n], dtype=torch.float32, device="cuda:0")
    env.t[:] = 0
    ne = min(args.export, n)
    frames = [[] for _ in range(ne)]
    t_reach = np.full(n, np.nan)
    obs = env.obs()
    rec = [None] * n                       # first episode result of every target pair
    for t in range(EPISODE_STEPS):
        with torch.no_grad():
            act = ac.actor(ac.norm(obs))
        qp = env.qpos[:ne].cpu().numpy()
        for k in range(ne):
            if rec[k] is None:
                frames[k].append([round(float(v), 4) for v in qp[k]])
        obs, _, done, infos = env.step(act)
        first = np.isnan(t_reach) & (env.hold >= 1).cpu().numpy() & np.array([x is None for x in rec])
        t_reach[first] = (t + 1) * DT_CTRL
        if infos is not None:
            for j, i in enumerate(torch.nonzero(done).squeeze(-1).tolist()):
                if rec[i] is None:
                    rec[i] = infos[j].cpu().numpy()
    assert all(x is not None for x in rec)
    infos = np.array(rec)
    rows = [[i[1], i[2], i[3], i[4], i[5], i[6], i[7], base["peak_opt"][k], t_reach[k], i[8]] for k, i in enumerate(infos)]
    rollouts = []
    for k in range(ne):
        _, e0, r0, e1, r1, peak, held, _, hit = (float(v) for v in infos[k])
        rollouts.append({"dt": DT_CTRL, "joints": names, "frames": frames[k],
                         "targets": [{"p": [float(v) for v in base["tgt_pos"][k][a]], "q": mat2quat(base["tgt_R"][k][a])}
                                     for a in range(2)],
                         "result": {"pos_err_mm": [round(e0 * 1000, 1), round(e1 * 1000, 1)],
                                    "rot_err_deg": [round(float(np.degrees(r0)), 1), round(float(np.degrees(r1)), 1)],
                                    "peak_torque_ratio": round(float(peak), 3), "success": bool(held), "collided": bool(hit)}})
    r = np.array(rows)
    pos = np.concatenate([r[:, 0], r[:, 2]]) * 1000
    rot = np.degrees(np.concatenate([r[:, 1], r[:, 3]]))
    arm_ok = (pos < POS_TOL * 1000) & (rot < np.degrees(ROT_TOL))
    print(f"checkpoint {args.ckpt} (iter {ck['iter']}), {n} target pairs, deterministic policy")
    print(f"  success (both arms within {POS_TOL * 1000:.0f} mm / {np.degrees(ROT_TOL):.1f} deg, held 0.5 s): {r[:, 5].mean():.1%}")
    print(f"  per-arm within tolerance at the end: {arm_ok.mean():.1%}")
    print(f"  position error mm : median {np.median(pos):.1f}  mean {pos.mean():.1f}  90% {np.percentile(pos, 90):.1f}")
    print(f"  rotation error deg: median {np.median(rot):.1f}  mean {rot.mean():.1f}  90% {np.percentile(rot, 90):.1f}")
    print(f"  time to first reach (successful episodes): median {np.nanmedian(r[:, 8]):.1f} s")
    print(f"  peak arm torque ratio over the whole motion: mean {r[:, 4].mean():.3f}  90% {np.percentile(r[:, 4], 90):.3f}  max {r[:, 4].max():.3f}")
    print(f"  episodes with any collision: {r[:, 9].mean():.1%}")
    print(f"  holding torque ratio at the end            : mean {r[:, 6].mean():.3f}")
    print(f"  optimiser static floor on the same targets : mean {r[:, 7].mean():.3f}")
    with open(os.path.join(HERE, "rollouts.json"), "w") as fh:
        json.dump(rollouts, fh)
    print("wrote rollouts.json with", len(rollouts), "rollouts")


if __name__ == "__main__":
    main()

"""Model-based baseline: for random target pairs, find the whole-body posture (base x/y/yaw, lift,
12 arm joints) that reaches both targets with the lowest static peak arm torque ratio
max_j |tau_gravity_j| / tau_peak_j.  This is the torque floor a policy can approach while holding.

    python3 baseline.py --n 200      -> baseline_targets.npz
"""
import argparse
import math
import os

import mujoco
import numpy as np
from scipy.optimize import minimize

from lift3_env import ARM_TPEAK, FINGER_OPEN, HERE, Lift3Batch

P_NORM = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    env = Lift3Batch(1, seed=args.seed)
    m, d, rng = env.m, env.scratch, env.rng
    v_arm = np.array([m.joint(m.dof_jntid[i]).dofadr[0] for i in range(m.nv)])[env.q_arm]  # dof == qpos index here
    lo = np.concatenate([[-10, -10, -10, 0.0], env.arm_lo])
    hi = np.concatenate([[10, 10, 10, env.lift_hi], env.arm_hi])

    def fk(x, collide=False):
        d.qpos[:] = 0.0
        d.qpos[0:3] = x[0:3]
        d.qpos[env.q_lift] = x[3]
        d.qpos[env.q_arm] = x[4:16]
        d.qpos[env.q_fing] = FINGER_OPEN
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        ratio = np.abs(d.qfrc_bias[v_arm]) / ARM_TPEAK
        return ratio

    out = {k: [] for k in ("tgt_pos", "tgt_R", "x_init", "x_opt", "peak_init", "peak_opt", "ok")}
    for n in range(args.n):
        lift, arms, p, R = env._sample_arm_config()
        dist = env.max_base_dist * math.sqrt(rng.random())
        ang, byaw = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)
        bx, by = dist * math.cos(ang), dist * math.sin(ang)
        c, s = math.cos(byaw), math.sin(byaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        tp = [Rz @ p[a] + np.array([bx, by, 0]) for a in range(2)]
        tR = [Rz @ R[a] for a in range(2)]
        x0 = np.concatenate([[bx, by, byaw, lift], arms])

        def cons(x):
            fk(x)
            e = []
            for a in range(2):
                e.append(d.site_xpos[env.site[a]] - tp[a])
                Rc = d.site_xmat[env.site[a]].reshape(3, 3)
                E = tR[a].T @ Rc
                e.append(0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]))
            return np.concatenate(e)

        def obj(x):
            r = fk(x)
            return float(np.sum(r ** P_NORM) ** (1.0 / P_NORM))

        peak0 = float(fk(x0).max())
        best_x, best_peak = x0, peak0
        res = minimize(obj, x0, method="SLSQP", bounds=list(zip(lo, hi)),
                       constraints=[{"type": "eq", "fun": cons}], options={"maxiter": 300, "ftol": 1e-7})
        x = np.clip(res.x, lo, hi)
        err = cons(x)
        peak = float(fk(x).max())
        mujoco.mj_collision(m, d)
        ok = np.abs(err[[0, 1, 2, 6, 7, 8]]).max() < 1e-3 and np.abs(err[[3, 4, 5, 9, 10, 11]]).max() < 5e-3 and d.ncon == 0
        if ok and peak < best_peak:
            best_x, best_peak = x, peak
        out["tgt_pos"].append(tp); out["tgt_R"].append(tR); out["x_init"].append(x0); out["x_opt"].append(best_x)
        out["peak_init"].append(peak0); out["peak_opt"].append(best_peak); out["ok"].append(bool(ok))
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{args.n}  sampled-posture peak {np.mean(out['peak_init']):.3f} -> optimised {np.mean(out['peak_opt']):.3f}"
                  f"   (optimiser valid {np.mean(out['ok']):.0%})", flush=True)
    np.savez(os.path.join(HERE, "baseline_targets.npz"), **{k: np.array(v) for k, v in out.items()})
    pi, po = np.array(out["peak_init"]), np.array(out["peak_opt"])
    print(f"static peak torque ratio over {args.n} target pairs:")
    print(f"  posture that generated the targets : mean {pi.mean():.3f}  median {np.median(pi):.3f}  max {pi.max():.3f}")
    print(f"  torque-optimal posture (baseline)  : mean {po.mean():.3f}  median {np.median(po):.3f}  max {po.max():.3f}")


if __name__ == "__main__":
    main()

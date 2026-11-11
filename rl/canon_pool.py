"""Replace every posture in config_pool.npz by an 'elegant' posture that reaches the SAME two grasp
poses (relative to each other), chosen to need little holding torque and little shoulder rotation.

Why: training targets and the guidance postures come from this pool. Postures drawn at random over
the joint ranges often have a shoulder joint turned far to the side for no reason, and the policy
copies that. Here each pair of grasp poses keeps its geometry, but the posture behind it is
re-solved: many inverse-kinematics attempts run in parallel on the GPU, slide along the set of
solutions towards low cost, and the best collision-free one is kept. (The chassis is free to turn,
so a sideways target does not need a turned shoulder.)

    ~/miniconda3/envs/rlwp/bin/python canon_pool.py        -> rewrites config_pool.npz (backup kept)
"""
import math
import os
import shutil
import time

import mujoco
import numpy as np
import torch

from dual_ik import POOL, DualIK
from lift3_env import ARM_TPEAK, FINGER_OPEN

SEEDS, PAIRS = 24, 256
YAW_W, CENTER_W = 0.15, 0.02


class Canon(DualIK):
    def __init__(self):
        super().__init__(seeds=SEEDS)
        m, D = self.m, lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.dev)
        self.mass, self.com = [], []
        for pre in ("left_link1", "right_link2"):
            ms, cs = [], []
            for k in range(1, 7):
                b = m.body(f"{pre}{k}")
                mm, cc = float(b.mass[0]), np.array(b.ipos) * float(b.mass[0])
                if k == 6:                                   # the two fingers ride on the gripper link (taken open)
                    for f in (7, 8):
                        fb = m.body(f"{pre}{f}")
                        ax = m.jnt_axis[fb.jntadr[0]]
                        mm += float(fb.mass[0])
                        cc += (np.array(fb.pos) + ax * FINGER_OPEN + np.array(fb.ipos)) * float(fb.mass[0])
                ms.append(mm)
                cs.append(D(cc / mm))
            self.mass.append(ms)
            self.com.append(cs)
        self.tpeak = D(ARM_TPEAK)
        self.mid, self.half = (self.lo[4:] + self.hi[4:]) / 2, (self.hi[4:] - self.lo[4:]) / 2

    def potential(self, x):
        """Gravitational potential energy of both arms (J); its gradient w.r.t. the arm joints is the holding torque."""
        n = x.shape[0]
        U = torch.zeros(n, dtype=torch.float64, device=self.dev)
        for a in range(2):
            p = self.base.expand(n, 3).clone()
            R = self.eye.expand(n, 3, 3)
            for k in range(6):
                p = p + R @ self.pos[a][k]
                q = x[:, 4 + a * 6 + k]
                R = R @ (self.eye + torch.sin(q)[:, None, None] * self.K[a * 6 + k] + (1 - torch.cos(q))[:, None, None] * self.K2[a * 6 + k])
                U = U + 9.81 * self.mass[a][k] * (p + R @ self.com[a][k])[:, 2]
        return U

    def cost_and_grad(self, x):
        with torch.enable_grad():
            xg = x.detach().clone().requires_grad_(True)
            tau = torch.autograd.grad(self.potential(xg).sum(), xg, create_graph=True)[0][:, 4:]
            ratio = tau / self.tpeak
            cost = (ratio.abs() ** 8).sum(1) ** 0.125 + YAW_W * (xg[:, 4] ** 2 + xg[:, 10] ** 2) \
                + CENTER_W * (((xg[:, 4:] - self.mid) / self.half) ** 2).sum(1)
            g = torch.autograd.grad(cost.sum(), xg)[0]
        return cost.detach(), g.detach(), ratio.detach().abs().max(1).values

    @torch.no_grad()
    def solve_batch(self, lift, arms, tp, tR, pool):
        """lift (P,), arms (P,12), tp (P,2,3), tR (P,2,3,3): poses in the generating posture's base frame."""
        Pn, S = lift.shape[0], SEEDS
        i = torch.randint(0, len(pool["lift"]), (Pn, S), generator=self.gen, device=self.dev)
        sp = pool["p"][i]                                                      # (P, S, 2, 3)
        ds, dt = sp[:, :, 0] - sp[:, :, 1], (tp[:, 0] - tp[:, 1])[:, None]
        yaw = torch.atan2(dt[..., 1], dt[..., 0]) - torch.atan2(ds[..., 1], ds[..., 0])
        c, s = torch.cos(yaw), torch.sin(yaw)
        ms, mt = sp.mean(2), tp.mean(1)[:, None]
        bx = mt[..., 0] - (c * ms[..., 0] - s * ms[..., 1])
        by = mt[..., 1] - (s * ms[..., 0] + c * ms[..., 1])
        x = torch.cat([bx[..., None], by[..., None], yaw[..., None], pool["lift"][i][..., None], pool["arms"][i]], -1)
        x[:, 0, :3] = 0.0                                                      # attempt 0 = the generating posture itself
        x[:, 0, 3] = lift
        x[:, 0, 4:] = arms
        x = x.reshape(Pn * S, 16)
        TP = tp[:, None].expand(Pn, S, 2, 3).reshape(-1, 2, 3)
        TR = tR[:, None].expand(Pn, S, 2, 3, 3).reshape(-1, 2, 3, 3)
        B = x.shape[0]
        I12 = torch.eye(12, dtype=torch.float64, device=self.dev)
        I16 = torch.eye(16, dtype=torch.float64, device=self.dev)
        eps, lam = 1e-6, 1e-3

        def err(xx, rep=1):
            p, R = self.fk(xx)
            tp_, tR_ = (TP, TR) if rep == 1 else (TP.repeat_interleave(rep, 0), TR.repeat_interleave(rep, 0))
            ep = tp_ - p
            er = 0.5 * torch.cross(R, tR_, dim=-2).sum(-1)
            return torch.cat([ep.reshape(-1, 6), 0.3 * er.reshape(-1, 6)], -1), ep, R, tR_

        for it in range(120):
            e = err(x)[0]
            xp = (x[:, None, :] + eps * I16[None]).reshape(-1, 16)
            J = ((err(xp, 16)[0].reshape(B, 16, 12) - e[:, None, :]) / eps).transpose(1, 2)
            Jp = J.transpose(1, 2) @ torch.linalg.inv(J @ J.transpose(1, 2) + lam * I12)       # (B, 16, 12)
            dx = -(Jp @ e[..., None])[..., 0]
            if it >= 40:                                                       # slide along the solution set towards low cost
                _, g, _ = self.cost_and_grad(x)
                g[:, :3] = 0.0
                dx = dx - 0.04 * ((I16 - Jp @ J) @ g[..., None])[..., 0].clamp(-0.5, 0.5)
            x = torch.minimum(torch.maximum(x + dx.clamp(-0.3, 0.3), self.lo), self.hi)
        for _ in range(15):                                                    # final pure-accuracy steps
            e = err(x)[0]
            xp = (x[:, None, :] + eps * I16[None]).reshape(-1, 16)
            J = ((err(xp, 16)[0].reshape(B, 16, 12) - e[:, None, :]) / eps).transpose(1, 2)
            dx = -(J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + 1e-6 * I12, e)[..., None])[..., 0]
            x = torch.minimum(torch.maximum(x + dx.clamp(-0.1, 0.1), self.lo), self.hi)
        _, ep, R, tR_ = err(x)
        pe = torch.linalg.norm(ep, dim=-1).max(1).values
        re = torch.arccos(((torch.einsum("naij,naij->na", R, tR_) - 1) / 2).clamp(-1, 1)).max(1).values
        cost, _, peak = self.cost_and_grad(x)
        cost = torch.where((pe < 1e-3) & (re < 5e-3), cost, torch.full_like(cost, 1e9))
        cost = cost.reshape(Pn, S)
        order = torch.argsort(cost, 1)[:, :5]
        return x.reshape(Pn, S, 16), cost, order


def main():
    t0 = time.time()
    ik = Canon()
    raw = np.load(POOL)
    P = {k: raw[k] for k in raw.files}
    if "canonical" in P:
        print("pool is already canonical"); return
    shutil.copy(POOL, POOL.replace(".npz", "_random.npz"))
    D = lambda v: torch.as_tensor(v, dtype=torch.float64, device=ik.dev)
    pool = {k: D(P[k]) for k in ("lift", "arms", "p")}
    n = len(P["lift"])
    m, d = ik.m, ik.d
    m.geom_margin[:] = 0.01
    v_arm = ik.q_arm
    sites = [m.site("tcp_l").id, m.site("tcp_r").id]
    new = {k: P[k].copy() for k in ("lift", "arms", "p", "R", "tau_g")}
    kept, peak_old, peak_new, yaw_old, yaw_new = 0, [], [], [], []

    def check(q16):
        d.qpos[:] = 0.0
        d.qpos[ik.q_lift] = q16[3]
        d.qpos[ik.q_arm] = q16[4:]
        d.qpos[ik.q_fing] = FINGER_OPEN
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        low = min(d.site_xpos[sites[0]][2], d.site_xpos[sites[1]][2])
        return d.ncon == 0 and low >= 0.06

    for b0 in range(0, n, PAIRS):
        sl = slice(b0, min(n, b0 + PAIRS))
        x, cost, order = ik.solve_batch(D(P["lift"][sl]), D(P["arms"][sl]), D(P["p"][sl]), D(P["R"][sl]), pool)
        x, cost, order = x.cpu().numpy(), cost.cpu().numpy(), order.cpu().numpy()
        for j in range(x.shape[0]):
            g = b0 + j
            peak_old.append(np.abs(P["tau_g"][g] / ARM_TPEAK).max())
            yaw_old.append(np.abs(P["arms"][g][[0, 6]]).mean())
            for c in order[j]:
                if cost[j, c] > 1e8:
                    break
                if check(x[j, c]):
                    new["lift"][g], new["arms"][g] = x[j, c, 3], x[j, c, 4:]
                    new["p"][g] = np.stack([d.site_xpos[s] for s in sites])
                    new["R"][g] = np.stack([d.site_xmat[s].reshape(3, 3) for s in sites])
                    new["tau_g"][g] = d.qfrc_bias[v_arm]
                    kept += 1
                    break
            peak_new.append(np.abs(new["tau_g"][g] / ARM_TPEAK).max())
            yaw_new.append(np.abs(new["arms"][g][[0, 6]]).mean())
        if (b0 // PAIRS) % 20 == 0:
            print(f"{sl.stop}/{n}  replaced {kept}  holding peak {np.mean(peak_old):.3f} -> {np.mean(peak_new):.3f}  "
                  f"|shoulder yaw| {np.degrees(np.mean(yaw_old)):.0f} -> {np.degrees(np.mean(yaw_new)):.0f} deg  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    P.update(new)
    P["canonical"] = np.array(1)
    np.savez(POOL, **P)
    print(f"done: {kept}/{n} postures replaced; mean holding peak {np.mean(peak_old):.3f} -> {np.mean(peak_new):.3f}; "
          f"mean |shoulder yaw| {np.degrees(np.mean(yaw_old)):.0f} -> {np.degrees(np.mean(yaw_new)):.0f} deg")


if __name__ == "__main__":
    main()

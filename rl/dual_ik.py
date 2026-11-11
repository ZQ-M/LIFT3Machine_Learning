"""Can both grippers be on their targets at the same time?

Both arms share one platform, one lift and one chassis, so the two targets are not independent.
This solver looks for one whole-body posture (chassis x / y / yaw, lift height, 12 arm joints) that
puts the left grasp point on the left target and the right grasp point on the right target, inside
the joint limits, and then checks that posture for collisions.

It runs many inverse-kinematics attempts in parallel on the GPU, each from a different start posture.
"""
import math
import os

import mujoco
import numpy as np
import torch

from lift3_env import FINGER_OPEN, HERE

XML = os.path.join(HERE, "lift3_vec.xml")
POOL = os.path.join(HERE, "config_pool.npz")
POS_OK, ROT_OK = 0.002, math.radians(1.0)          # an attempt counts as solved within 2 mm and 1 degree


class DualIK:
    def __init__(self, device="cuda:0", seeds=768, iters=80):
        self.dev, self.B, self.iters = torch.device(device), seeds, iters
        m = self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(m)
        D = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.dev)
        arm = [f"left_joint1{i}" for i in range(1, 7)] + [f"right_joint2{i}" for i in range(1, 7)]
        self.q_arm = np.array([m.joint(n).qposadr[0] for n in arm])
        self.q_lift = m.joint("joint5").qposadr[0]
        self.q_fing = np.array([m.joint(n).qposadr[0] for n in ("left_joint17", "left_joint18", "right_joint27", "right_joint28")])
        self.lo = D(np.concatenate([[-1e9, -1e9, -1e9, 0.0], [m.joint(n).range[0] for n in arm]]))
        self.hi = D(np.concatenate([[1e9, 1e9, 1e9, m.joint("joint5").range[1]], [m.joint(n).range[1] for n in arm]]))
        self.base = D([0.0, 0.0, m.body("base_link").pos[2]]) + D(m.body("body5").pos)
        self.pos = [[D(m.body(f"{pre}{k}").pos) for k in range(1, 7)] for pre in ("left_link1", "right_link2")]
        K = []
        for n in arm:
            ax = m.jnt_axis[m.joint(n).id]
            K.append(D([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]))
        self.K, self.K2 = K, [k @ k for k in K]
        self.tcp = [D(m.site(s).pos) for s in ("tcp_l", "tcp_r")]
        P = np.load(POOL)
        self.pool = {k: D(P[k]) for k in ("lift", "arms", "p")}
        self.eye = torch.eye(3, dtype=torch.float64, device=self.dev)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(0)

    def fk(self, x):
        """x: (n, 16) = base x, y, yaw, lift, 12 arm joints -> grasp point positions (n, 2, 3) and rotations."""
        n = x.shape[0]
        c, s = torch.cos(x[:, 2]), torch.sin(x[:, 2])
        z, o = torch.zeros_like(c), torch.ones_like(c)
        Rz = torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(n, 3, 3)
        pb = torch.stack([x[:, 0], x[:, 1], z], -1)
        ps, Rs = [], []
        for a in range(2):
            p = self.base.expand(n, 3).clone()
            p[:, 2] += x[:, 3]
            R = self.eye.expand(n, 3, 3)
            for k in range(6):
                p = p + R @ self.pos[a][k]
                q = x[:, 4 + a * 6 + k]
                R = R @ (self.eye + torch.sin(q)[:, None, None] * self.K[a * 6 + k] + (1 - torch.cos(q))[:, None, None] * self.K2[a * 6 + k])
            p = p + R @ self.tcp[a]
            ps.append(torch.einsum("nij,nj->ni", Rz, p) + pb)
            Rs.append(Rz @ R)
        return torch.stack(ps, 1), torch.stack(Rs, 1)

    def err(self, x, tp, tR):
        p, R = self.fk(x)
        ep = tp[None] - p                                                    # (n, 2, 3)
        er = 0.5 * torch.cross(R, tR[None].expand_as(R), dim=-2).sum(-1)     # (n, 2, 3) rotation vector, world frame
        return torch.cat([ep.reshape(-1, 6), 0.3 * er.reshape(-1, 6)], -1)   # 0.3 m per rad balances the two

    @torch.no_grad()
    def solve(self, targets):
        """targets: [{"p": [x, y, z], "q": [x, y, z, w]}, ...] for the left and right gripper."""
        D = lambda v: torch.tensor(v, dtype=torch.float64, device=self.dev)
        tp = torch.stack([D(t["p"]) for t in targets])
        tR = []
        for t in targets:
            x, y, z, w = t["q"]
            tR.append(D([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]))
        tR = torch.stack(tR)
        B = self.B
        # start postures: random feasible arm postures, with the chassis placed so the posture's own grasp
        # points lie roughly over the targets
        i = torch.randint(0, len(self.pool["lift"]), (B,), generator=self.gen, device=self.dev)
        sp = self.pool["p"][i]
        ds, dt = sp[:, 0] - sp[:, 1], tp[0] - tp[1]
        yaw = math.atan2(dt[1], dt[0]) - torch.atan2(ds[:, 1], ds[:, 0])
        c, s = torch.cos(yaw), torch.sin(yaw)
        mid_s = sp.mean(1)
        mid_t = tp.mean(0)
        bx = mid_t[0] - (c * mid_s[:, 0] - s * mid_s[:, 1])
        by = mid_t[1] - (s * mid_s[:, 0] + c * mid_s[:, 1])
        x = torch.cat([bx[:, None], by[:, None], yaw[:, None], self.pool["lift"][i][:, None], self.pool["arms"][i]], -1)
        eps, lam = 1e-6, 1e-3
        I12 = torch.eye(12, dtype=torch.float64, device=self.dev)
        for _ in range(self.iters):
            e = self.err(x, tp, tR)
            xp = x[:, None, :] + eps * torch.eye(16, dtype=torch.float64, device=self.dev)[None]
            J = ((self.err(xp.reshape(-1, 16), tp, tR).reshape(B, 16, 12) - e[:, None, :]) / eps).transpose(1, 2)   # (B, 12, 16)
            dx = -torch.einsum("bji,bj->bi", J, torch.linalg.solve(J @ J.transpose(1, 2) + lam * I12, e))
            x = torch.minimum(torch.maximum(x + dx.clamp(-0.3, 0.3), self.lo), self.hi)
        p, R = self.fk(x)
        pe = torch.linalg.norm(tp[None] - p, dim=-1).max(1).values
        tr = torch.einsum("naij,aij->na", R, tR)
        re = torch.arccos(((tr - 1) / 2).clamp(-1, 1)).max(1).values
        ok = (pe < POS_OK) & (re < ROT_OK)
        score = pe + 0.3 * re
        best = int(torch.argmin(score))
        out = {"reachable": bool(ok.any()), "collision_free": False, "attempts": B, "solved": int(ok.sum()),
               "best_pos_err_mm": round(float(pe[best]) * 1000, 1), "best_rot_err_deg": round(math.degrees(float(re[best])), 1),
               "target_distance_m": round(float(torch.linalg.norm(dt)), 3), "solution": None}
        if not out["reachable"]:
            return out
        # collision check on the CPU model for the solved postures
        sol = x[ok].cpu().numpy()[:60]
        first = None
        for q in sol:
            d = self.d
            d.qpos[:] = 0.0
            d.qpos[0:3] = q[0:3]
            d.qpos[self.q_lift] = q[3]
            d.qpos[self.q_arm] = q[4:]
            d.qpos[self.q_fing] = FINGER_OPEN
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_collision(self.m, d)
            if first is None:
                first = d.qpos.copy()
            if d.ncon == 0:
                out["collision_free"] = True
                out["solution"] = [round(float(v), 4) for v in d.qpos]
                return out
        out["solution"] = [round(float(v), 4) for v in first]
        return out


if __name__ == "__main__":
    import time
    ik = DualIK()
    P = np.load(POOL)
    R2q = lambda R: (lambda q: [q[1], q[2], q[3], q[0]])((lambda q: (mujoco.mju_mat2Quat(q, R.ravel()), q)[1])(np.zeros(4)))
    n_ok = 0
    t = time.time()
    for k in range(20):                      # targets that certainly have a solution
        r = ik.solve([{"p": P["p"][k][a].tolist(), "q": R2q(P["R"][k][a])} for a in range(2)])
        n_ok += r["reachable"] and r["collision_free"]
    print(f"known-solvable pairs found solvable: {n_ok}/20, {(time.time() - t) / 20:.2f} s per query")
    far = [{"p": [0.5, 1.2, 0.8], "q": [0, 0, 0, 1]}, {"p": [0.5, -1.2, 0.8], "q": [0, 0, 0, 1]}]
    print("targets 2.4 m apart:", {k: v for k, v in ik.solve(far).items() if k != "solution"})
    near = [{"p": [0.55, 0.2, 0.75], "q": [0, 0, 0, 1]}, {"p": [0.55, -0.2, 0.75], "q": [0, 0, 0, 1]}]
    print("default targets:", {k: v for k, v in ik.solve(near).items() if k != "solution"})

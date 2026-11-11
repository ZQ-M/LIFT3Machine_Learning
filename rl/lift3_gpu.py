"""LIFT3 reaching environment fully on the GPU: physics by mujoco_warp, everything else in torch.
Run with the `rlwp` conda environment (mujoco_warp + warp + torch are installed there).

Same task and action layout as lift3_vec.py:
    0:4 wheel SPEED commands, 4 lift POSITION, 5:17 arm joint POSITIONS, 17:19 gripper openings.
    Position outputs span each joint's range (-1 = lower limit, +1 = upper limit).
"""
import math
import os
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from lift3_env import (ACT_DIM, ARM_TPEAK, ARM_VMAX, DECIMATION, DT_CTRL, EPISODE_STEPS, FINGER_OPEN, HERE,
                       LIFT_VMAX, POS_TOL, ROT_TOL, TARGET_LEAD, WHEEL_TMAX, WHEEL_VMAX)

XML = os.path.join(HERE, "lift3_vec.xml")
POOL = os.path.join(HERE, "config_pool.npz")
OBS_DIM = 135
ARM_AMAX = (6.0, 6.0, 6.0, 12.0, 12.0, 12.0) * 2   # rad/s^2: the position target may accelerate at most this fast
TRACK_GAIN = 8.0         # 1/s: how quickly the smoothed target closes in on the commanded position
WHEEL_DEADBAND = 0.05
KP_LEN = 0.12            # keypoints sit this far along each gripper axis
wp.config.quiet = True


def quat2mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def rotz(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(yaw.shape + (3, 3))


class Lift3GPU:
    def __init__(self, N, seed=0, max_base_dist=1.5, device="cuda:0"):
        self.N, self.dev = N, torch.device(device)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        pool = np.load(POOL)
        self.pool = {k: torch.tensor(pool[k], dtype=torch.float32, device=self.dev) for k in pool.files if k not in ("n_uniform", "roomy")}
        self.roomy_idx = torch.as_tensor(np.nonzero(pool["roomy"])[0], dtype=torch.long, device=self.dev)   # start postures with 3 cm clearance
        self.n_uniform = int(pool["n_uniform"])      # pool entries after this index are the low-gripper ones
        mjm = self.mjm = mujoco.MjModel.from_xml_path(XML)
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        wp.set_device(device)
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mjd, nworld=N, nconmax=16, njmax=160)
        mjw.step(self.m, self.d)                       # compile / load kernels
        with wp.ScopedCapture() as cap:                # one physics step as a CUDA graph
            mjw.step(self.m, self.d)
        self.graph = cap.graph
        # zero-copy torch views of the simulator state
        self.qpos, self.qvel = wp.to_torch(self.d.qpos), wp.to_torch(self.d.qvel)
        self.ctrl, self.sens = wp.to_torch(self.d.ctrl), wp.to_torch(self.d.sensordata)
        self.warm, self.time = wp.to_torch(self.d.qacc_warmstart), wp.to_torch(self.d.time)

        T = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.dev)
        jq = lambda n: mjm.joint(n).qposadr[0]
        arm_names = [f"left_joint1{i}" for i in range(1, 7)] + [f"right_joint2{i}" for i in range(1, 7)]
        fing = ("left_joint17", "left_joint18", "right_joint27", "right_joint28")
        L = torch.long
        self.q_lift = int(jq("joint5"))
        self.q_arm = T([jq(n) for n in arm_names], L)
        self.q_fing = T([jq(n) for n in fing], L)
        self.arm_lo = T([mjm.joint(n).range[0] for n in arm_names])
        self.arm_hi = T([mjm.joint(n).range[1] for n in arm_names])
        self.lift_hi = float(mjm.joint("joint5").range[1])
        self.u_lift = int(mjm.actuator("joint5").id)
        self.u_arm = T([mjm.actuator(n).id for n in arm_names], L)
        self.u_fing = T([mjm.actuator(n).id for n in fing], L)
        self.u_wheel = T([mjm.actuator(f"wheel{k}").id for k in range(1, 5)], L)
        sa = lambda n: int(mjm.sensor(n).adr[0])
        self.s_arm = T([sa("frc_" + n) for n in arm_names], L)
        self.s_wheel = T([sa(f"frc_wheel{k}") for k in range(1, 5)], L)
        self.s_touch = T([mjm.sensor(i).adr[0] for i in range(mjm.nsensor) if mjm.sensor(i).name.startswith("touch_")], L)
        self.s_pos = [sa("tcp_l_pos"), sa("tcp_r_pos")]
        self.s_quat = [sa("tcp_l_quat"), sa("tcp_r_quat")]
        self.arm_vmax, self.arm_tpeak, self.arm_amax = T(ARM_VMAX), T(ARM_TPEAK), T(ARM_AMAX)
        self.arm_kp = T([mjm.actuator(n).gainprm[0] for n in arm_names])
        self.max_base_dist = max_base_dist
        # --- kinematic chain for batched forward kinematics (used to build curriculum targets)
        bp = lambda n: T(mjm.body(n).pos)
        self.fk_base = T([0.0, 0.0, mjm.body("base_link").pos[2]]) + bp("body5")
        self.fk_pos = [[bp(f"{pre}{k}") for k in range(1, 7)] for pre in ("left_link1", "right_link2")]
        self.fk_axis = [[T(mjm.jnt_axis[mjm.joint(n).id]) for n in arm_names[a * 6:a * 6 + 6]] for a in range(2)]
        self.fk_tcp = [T(mjm.site(s).pos) for s in ("tcp_l", "tcp_r")]
        # --- curriculum: targets are a fraction alpha of the way (in joint space, and in base displacement)
        # from the start configuration to a random feasible one; alpha_max grows as the policy succeeds
        self.alpha_max = 0.15
        # guidance: every training target comes from a known posture (lift, arm joints, chassis pose).
        # While guide_w > 0 the policy is rewarded for commanding that posture, which teaches the
        # target -> joint position mapping directly; the weight is reduced later so the torque terms decide.
        self.guide_w = 1.0
        self.a_star = torch.zeros(N, 13, device=self.dev)
        self.base_star = torch.zeros(N, 3, device=self.dev)
        self.cur_good, self.cur_n = 0.0, 0
        self.arm_mid, self.arm_half = (self.arm_lo + self.arm_hi) / 2, (self.arm_hi - self.arm_lo) / 2
        self.joint_w = T([3.0, 2.0, 2.0, 1.0, 1.0, 1.0] * 2)      # joint travel cost: shoulder joints cost most
        self.live = False                             # live mode: never reset, keep running on moving targets
        self.last = {}

        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.dev)
        self.t = z(N, dt=L)
        self.arm_tgt, self.lift_tgt = z(N, 12), z(N)
        self.arm_vcmd = z(N, 12)                      # smoothed joint speed command
        self.timeout = z(N, dt=torch.bool)
        self.collided = z(N, dt=torch.bool)          # any contact so far in this episode
        self.prev_act = z(N, ACT_DIM)
        self.tgt_pos, self.tgt_R = z(N, 2, 3), z(N, 2, 3, 3)
        self.tcp_pos, self.tcp_R = z(N, 2, 3), z(N, 2, 3, 3)
        self.peak, self.ep_ret = z(N), z(N)
        self.hold = z(N, dt=L)
        self.reset(torch.arange(N, device=self.dev))
        self.t = torch.randint(0, EPISODE_STEPS, (N,), generator=self.gen, device=self.dev)   # spread episode ends over time

    def _rand(self, n):
        return torch.rand(n, generator=self.gen, device=self.dev)

    def _randint(self, hi, n):
        return torch.randint(0, hi, (n,), generator=self.gen, device=self.dev)

    def reset(self, idx):
        n, P = len(idx), self.pool
        if n == 0:
            return
        use_pool = self._rand(n) < 0.5
        i_p, i_h = self.roomy_idx[self._randint(len(self.roomy_idx), n)], self._randint(len(P["h_lift"]), n)
        lift = torch.where(use_pool, P["lift"][i_p], P["h_lift"][i_h])
        arms = torch.where(use_pool[:, None], P["arms"][i_p], P["h_arms"][i_h])
        p0 = torch.where(use_pool[:, None, None], P["p"][i_p], P["h_p"][i_h])
        R0 = torch.where(use_pool[:, None, None, None], P["R"][i_p], P["h_R"][i_h])
        yaw0 = (self._rand(n) * 2 - 1) * math.pi
        q = torch.zeros(n, self.qpos.shape[1], device=self.dev)
        q[:, 2] = yaw0
        q[:, self.q_lift] = lift
        q[:, self.q_arm] = arms
        q[:, self.q_fing] = FINGER_OPEN
        self.qpos[idx] = q
        self.qvel[idx] = 0.0
        self.warm[idx] = 0.0
        self.time[idx] = 0.0
        Rz0 = rotz(yaw0)
        self.tcp_pos[idx] = torch.einsum("nij,naj->nai", Rz0, p0)
        self.tcp_R[idx] = torch.einsum("nij,najk->naik", Rz0, R0)
        # start with the PD targets offset by gravity torque / stiffness, so the arm does not sag at t = 0
        tau_g = torch.where(use_pool[:, None], P["tau_g"][i_p], P["h_tau_g"][i_h])
        self.arm_tgt[idx] = torch.minimum(torch.maximum(arms + tau_g / self.arm_kp, self.arm_lo), self.arm_hi)
        self.arm_vcmd[idx] = 0.0
        self.lift_tgt[idx] = lift
        j = self._randint(len(P["lift"]), n)
        low = self.n_uniform + self._randint(len(P["lift"]) - self.n_uniform, n)
        j = torch.where(self._rand(n) < 0.3, low, j)             # 30% of targets have a gripper near the floor
        alpha = torch.maximum(self._rand(n), self._rand(n)) * self.alpha_max      # weighted towards the harder end
        if self.alpha_max >= 1.0:                                # at full difficulty most targets are whole pool postures,
            alpha = torch.where(self._rand(n) < 0.7, torch.ones_like(alpha), alpha)      # which are known collision-free
        t_lift = lift + alpha * (P["lift"][j] - lift)
        t_arms = arms + alpha[:, None] * (P["arms"][j] - arms)
        tp, tR = self.fk(t_lift, t_arms)
        # in-between postures are not collision-checked: if the two grasp points come out too close together
        # or too low, use the whole pool posture instead
        bad = (torch.linalg.norm(tp[:, 0] - tp[:, 1], dim=-1) < 0.15) | (tp[:, :, 2].min(1).values < 0.06)
        alpha = torch.where(bad, torch.ones_like(alpha), alpha)
        t_lift = torch.where(bad, P["lift"][j], t_lift)
        t_arms = torch.where(bad[:, None], P["arms"][j], t_arms)
        tp = torch.where(bad[:, None, None], P["p"][j], tp)
        tR = torch.where(bad[:, None, None, None], P["R"][j], tR)
        dist = alpha * self.max_base_dist * torch.sqrt(self._rand(n))
        ang = (self._rand(n) * 2 - 1) * math.pi
        off = torch.stack([dist * torch.cos(ang), dist * torch.sin(ang), torch.zeros_like(dist)], -1)
        Rz_yaw = yaw0 + alpha * (self._rand(n) * 2 - 1) * math.pi
        Rz = rotz(Rz_yaw)
        self.tgt_pos[idx] = torch.einsum("nij,naj->nai", Rz, tp) + off[:, None, :]
        self.tgt_R[idx] = torch.einsum("nij,najk->naik", Rz, tR)
        self.a_star[idx] = self.hold_action(t_lift, t_arms)[:, 4:17]
        self.base_star[idx] = torch.stack([off[:, 0], off[:, 1], yaw0 + (Rz_yaw - yaw0)], -1)
        self.t[idx] = 0
        self.prev_act[idx] = self.hold_action(lift, arms)
        self.peak[idx] = 0.0
        self.ep_ret[idx] = 0.0
        self.hold[idx] = 0
        self.collided[idx] = False

    def hold_action(self, lift, arms):
        """The action that commands 'stay exactly here' (wheels stopped, grippers open)."""
        a = torch.zeros(lift.shape[0], ACT_DIM, device=self.dev)
        a[:, 4] = lift / self.lift_hi * 2.0 - 1.0
        a[:, 5:17] = (arms - self.arm_mid) / self.arm_half
        a[:, 17:19] = 1.0
        return a

    def fk(self, lift, arms):
        """TCP poses (n, 2, 3) and (n, 2, 3, 3) for the base at the origin with zero yaw."""
        n = lift.shape[0]
        ps, Rs = [], []
        for a in range(2):
            p = self.fk_base.expand(n, 3).clone()
            p[:, 2] += lift
            R = torch.eye(3, device=self.dev).expand(n, 3, 3)
            for k in range(6):
                p = p + torch.einsum("nij,j->ni", R, self.fk_pos[a][k])
                ax, q = self.fk_axis[a][k], arms[:, a * 6 + k]
                K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], device=self.dev)
                Rq = torch.eye(3, device=self.dev) + torch.sin(q)[:, None, None] * K + (1 - torch.cos(q))[:, None, None] * (K @ K)
                R = R @ Rq
            ps.append(p + torch.einsum("nij,j->ni", R, self.fk_tcp[a]))
            Rs.append(R)
        return torch.stack(ps, 1), torch.stack(Rs, 1)

    def errors(self):
        dist = torch.linalg.norm(self.tgt_pos - self.tcp_pos, dim=-1)
        tr = torch.einsum("naij,naij->na", self.tcp_R, self.tgt_R)
        ang = torch.arccos(torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0))
        return dist, ang

    def obs(self):
        q, v = self.qpos, self.qvel
        yaw = q[:, 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        Rb = rotz(yaw)
        pb = torch.stack([q[:, 0], q[:, 1], torch.zeros_like(yaw)], -1)
        vb = torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)
        p = torch.einsum("nji,naj->nai", Rb, self.tcp_pos - pb[:, None])
        R = torch.einsum("nji,najk->naik", Rb, self.tcp_R)
        tp = torch.einsum("nji,naj->nai", Rb, self.tgt_pos - pb[:, None])
        tR = torch.einsum("nji,najk->naik", Rb, self.tgt_R)
        e = tp - p
        n = torch.linalg.norm(e, dim=-1, keepdim=True)
        e_dir = e / n.clamp_min(1e-6) * n.clamp_max(1.0)
        e_loc = torch.einsum("naji,naj->nai", R, e)
        e_loc = torch.where(n < 0.2, e_loc * 5.0, e_loc / n.clamp_min(1e-6))
        r6 = lambda M: torch.cat([M[..., :, 0], M[..., :, 1]], -1)
        R_rel = torch.einsum("naji,najk->naik", R, tR)                                  # target rotation seen from the gripper
        per_arm = torch.cat([p, r6(R), tp.clamp(-3.0, 3.0), r6(tR), e_dir, e_loc, r6(R_rel)], -1).reshape(self.N, -1)
        return torch.cat([
            q[:, self.q_arm], v[:, self.q_arm] * 0.2,
            (q[:, self.q_lift] / self.lift_hi)[:, None], (v[:, self.q_lift] * 10.0)[:, None],
            (self.lift_tgt / self.lift_hi)[:, None],
            q[:, self.q_fing[[0, 2]]] / FINGER_OPEN, vb, self.arm_tgt - q[:, self.q_arm],
            per_arm, self.prev_act, self.arm_vcmd / self.arm_vmax], -1)

    def step(self, act):
        act = act.clamp(-1.0, 1.0)
        q_arm = self.qpos[:, self.q_arm]
        lift_des = (act[:, 4] + 1.0) * 0.5 * self.lift_hi                    # commanded lift height
        self.lift_tgt = self.lift_tgt + (lift_des - self.lift_tgt).clamp(-LIFT_VMAX * DT_CTRL, LIFT_VMAX * DT_CTRL)
        # The policy commands joint POSITIONS. The PD target does not jump there: it travels towards the
        # commanded position with limited speed and acceleration and brakes in time to stop on it, so a
        # steady command gives a perfectly still arm and a changed command gives a smooth move.
        e = (self.arm_mid + act[:, 5:17] * self.arm_half) - self.arm_tgt
        ea = e.abs()
        v_des = torch.sign(e) * torch.minimum(torch.minimum(TRACK_GAIN * ea, torch.sqrt(2.0 * self.arm_amax * ea)), self.arm_vmax)
        dv = v_des - self.arm_vcmd
        lim = self.arm_amax * DT_CTRL
        self.arm_vcmd = self.arm_vcmd + torch.minimum(torch.maximum(dv, -lim), lim)
        tgt = self.arm_tgt + self.arm_vcmd * DT_CTRL
        tgt = torch.minimum(torch.maximum(tgt, q_arm - TARGET_LEAD), q_arm + TARGET_LEAD)
        self.arm_tgt = torch.minimum(torch.maximum(tgt, self.arm_lo), self.arm_hi)
        u = self.ctrl
        # wheel outputs smaller than 5% count as exactly zero, so "stand still" is easy to command and a
        # tiny leftover output cannot make the chassis creep
        w = act[:, 0:4]
        w = torch.sign(w) * (w.abs() - WHEEL_DEADBAND).clamp_min(0.0) / (1.0 - WHEEL_DEADBAND)
        u[:, self.u_wheel] = w * WHEEL_VMAX
        u[:, self.u_lift] = self.lift_tgt
        u[:, self.u_arm] = self.arm_tgt
        g = (act[:, 17:19] + 1.0) * 0.5 * FINGER_OPEN
        u[:, self.u_fing] = g[:, [0, 0, 1, 1]]
        ratio = torch.zeros(self.N, 12, device=self.dev)
        wt2 = torch.zeros(self.N, device=self.dev)
        contact = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        for _ in range(DECIMATION):
            wp.capture_launch(self.graph)
            s = self.sens
            ratio = torch.maximum(ratio, s[:, self.s_arm].abs())
            wt2 = wt2 + (s[:, self.s_wheel] ** 2).sum(1)
            contact = contact | (s[:, self.s_touch] > 0.0).any(1)
        ratio = ratio / self.arm_tpeak
        s = self.sens
        for a in range(2):
            self.tcp_pos[:, a] = s[:, self.s_pos[a]:self.s_pos[a] + 3]
            self.tcp_R[:, a] = quat2mat(s[:, self.s_quat[a]:self.s_quat[a] + 4])
        # ---- reward
        cur_peak = ratio.max(1).values
        new_peak = torch.maximum(self.peak, cur_peak)
        d_peak = new_peak - self.peak
        self.peak = new_peak
        dist, ang = self.errors()
        # Keypoint distance: mean distance between 4 matching points (origin + one along each axis) of the
        # gripper frame and the target frame.  One number that is zero only when position AND orientation
        # both match, so the policy cannot settle for the right position with the wrong orientation.
        d0 = self.tgt_pos - self.tcp_pos
        dk = d0[..., None] + KP_LEN * (self.tgt_R - self.tcp_R)
        d_kp = (dist + torch.linalg.norm(dk, dim=-2).sum(-1)) / 4.0
        r = (0.5 * (1 - torch.tanh(d_kp / 0.5)) + 1.0 * (1 - torch.tanh(d_kp / 0.08))
             + 0.6 * (1 - torch.tanh(d_kp / 0.02))).sum(1)
        inside = (dist < POS_TOL) & (ang < ROT_TOL)
        want_open = torch.where((dist < 0.02) & (ang < 0.15), 0.0, 1.0)
        r = r - 0.2 * ((act[:, 17:19] + 1.0) * 0.5 - want_open).abs().sum(1)
        r = r + 0.5 * inside.sum(1)
        ok = inside.all(1)
        r = r + 1.0 * ok
        self.hold = torch.where(ok, self.hold + 1, torch.zeros_like(self.hold))
        r = r - (1.5 * cur_peak ** 2 + 0.3 * (ratio ** 2).mean(1) + 100.0 * d_peak)
        if self.guide_w > 0 and not self.live:
            rms = torch.sqrt(((act[:, 4:17] - self.a_star) ** 2).mean(1))
            dxy = torch.linalg.norm(self.qpos[:, 0:2] - self.base_star[:, 0:2], dim=1)
            dyaw = torch.atan2(torch.sin(self.qpos[:, 2] - self.base_star[:, 2]), torch.cos(self.qpos[:, 2] - self.base_star[:, 2])).abs()
            r = r + self.guide_w * (1.5 * (1 - torch.tanh(rms / 0.15)) + 0.5 * (1 - torch.tanh(dxy / 0.3)) + 0.3 * (1 - dyaw / math.pi))
        r = r - 0.05 * ((act - self.prev_act) ** 2).sum(1)
        # posture preference: keep the shoulder (root) joints near straight ahead; the chassis can turn instead
        r = r - 0.12 * (q_arm[:, 0] ** 2 + q_arm[:, 6] ** 2)
        r = r - 0.08 * (self.joint_w * (self.arm_vcmd / self.arm_vmax) ** 2).sum(1)
        # once both grippers are close, any remaining joint motion costs in proportion to its speed: hold still
        r = r - 2.0 * torch.exp(-d_kp.max(1).values / 0.05) * (self.arm_vcmd.abs() / self.arm_vmax).sum(1)     # joint travel: prefer short, calm motions
        r = r - 0.01 * wt2 / (DECIMATION * WHEEL_TMAX ** 2)
        # chassis should come to rest once both grippers are close to their targets
        vb2 = self.qvel[:, 0] ** 2 + self.qvel[:, 1] ** 2 + 0.3 * self.qvel[:, 2] ** 2
        r = r - 1.0 * vb2 * torch.exp(-d_kp.max(1).values / 0.1)
        r = r - 1.5 * torch.exp(-d_kp.max(1).values / 0.05) * (self.qvel[:, 0].abs() + self.qvel[:, 1].abs() + 0.3 * self.qvel[:, 2].abs())
        r = (r - 5.0 * contact) * 0.2                  # every step in contact costs most of that step's reward
        self.prev_act = act
        if self.live:
            self.last = {"dist": dist, "ang": ang, "ratio": ratio, "contact": contact}
            return self.obs(), r, None, None
        self.t += 1
        self.ep_ret += r
        self.timeout = self.t >= EPISODE_STEPS
        self.collided = self.collided | contact
        done = self.timeout
        idx = torch.nonzero(done).squeeze(-1)
        infos = None
        if len(idx):
            # curriculum: when most finished episodes ended close to both targets, make targets harder
            self.cur_good += float((d_kp[idx].max(1).values < 0.03).sum())
            self.cur_n += len(idx)
            if self.cur_n >= 2000:
                if self.cur_good / self.cur_n > 0.5:
                    self.alpha_max = min(1.0, self.alpha_max + 0.05)
                self.cur_good, self.cur_n = 0.0, 0
            infos = torch.stack([self.ep_ret[idx], dist[idx, 0], ang[idx, 0], dist[idx, 1], ang[idx, 1], self.peak[idx],
                                 (self.hold[idx] >= 25).float(), cur_peak[idx], self.collided[idx].float()], -1)
            self.reset(idx)
        return self.obs(), r, done, infos


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    env = Lift3GPU(N)
    o = env.obs()
    assert o.shape == (N, OBS_DIM), o.shape
    for _ in range(5):
        env.step(torch.rand(N, ACT_DIM, device=env.dev) * 2 - 1)
    torch.cuda.synchronize(); wp.synchronize()
    t = time.time()
    for _ in range(100):
        o, r, d, inf = env.step(torch.rand(N, ACT_DIM, device=env.dev) * 2 - 1)
    torch.cuda.synchronize(); wp.synchronize()
    dt = time.time() - t
    print(f"N={N}: {N * 100 / dt:.0f} env steps/s; reward mean {r.mean().item():.3f}; obs finite {torch.isfinite(o).all().item()}; "
          f"absmax {o.abs().max().item():.2f}; contacts in use {int(wp.to_torch(env.d.nacon)[0])} of {env.d.naconmax}")

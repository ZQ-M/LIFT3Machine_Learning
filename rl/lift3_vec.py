"""Batched LIFT3 reaching environment: N robots stepped together by MuJoCo's multi-threaded
rollout (C threads, no Python in the physics loop).  Same task, actions and rewards as lift3_env.py.

Action (19, each in [-1, 1]):
    0:4    wheel speed commands            -> x WHEEL_VMAX rad/s (native torque-limited velocity servos)
    4      lift target velocity            -> x LIFT_VMAX m/s
    5:17   arm joint target velocities     -> x ARM_VMAX rad/s (integrated into PD position targets)
    17,18  left / right gripper (-1 closed .. +1 open)
"""
import math
import os
import time

import mujoco
import numpy as np
from mujoco import rollout

from lift3_env import (ACT_DIM, ARM_TPEAK, ARM_VMAX, DECIMATION, DT_CTRL, EPISODE_STEPS, FINGER_OPEN, HERE,
                       LIFT_VMAX, POS_TOL, ROT_TOL, TARGET_LEAD, WHEEL_TMAX, WHEEL_VMAX, Lift3Batch)

XML = os.path.join(HERE, "lift3_vec.xml")
POOL = os.path.join(HERE, "config_pool.npz")
OBS_DIM = 111


def build_pool(n=40000, seed=7):
    """Collision-free (lift, arm) configurations with their TCP poses in the base frame."""
    env = Lift3Batch(1, seed=seed)
    env.m.geom_margin[:] = 0.01        # two shapes closer than 1 cm count as touching: sampled poses keep clearance
    lift, arms, p, R, tg, roomy = [], [], [], [], [], []
    def is_roomy():                          # is the posture just sampled at least 3 cm clear of everything?
        env.m.geom_margin[:] = 0.03
        mujoco.mj_collision(env.m, env.scratch)
        ok = True
        for con in env.scratch.contact[:env.scratch.ncon]:
            pair = {env.m.geom(con.geom1).name, env.m.geom(con.geom2).name}
            # the upper arm's shoulder end always sits about 1 cm above the platform: that is not a tight spot
            if pair in ({"body5_c1", "left_link12_c0"}, {"body5_c1", "right_link22_c0"}) and con.dist > 0.005:
                continue
            ok = False
        env.m.geom_margin[:] = 0.01
        return ok
    v_arm = env.q_arm                       # dof index == qpos index for this model
    def gravity(a, b):
        ds = env.scratch
        ds.qpos[:] = 0.0; ds.qpos[env.q_lift] = a; ds.qpos[env.q_arm] = b; ds.qpos[env.q_fing] = FINGER_OPEN
        ds.qvel[:] = 0.0
        mujoco.mj_forward(env.m, ds)
        return ds.qfrc_bias[v_arm].copy()
    for _ in range(n):
        # 80% natural postures (shoulder yaw within about +-17 deg, 1 sigma), 20% anything in the joint range
        a, b, c, d = env._sample_arm_config(0.3 if env.rng.random() < 0.8 else None)
        roomy.append(is_roomy())
        lift.append(a); arms.append(b); p.append(c); R.append(d); tg.append(gravity(a, b))
    # extra configurations with at least one gripper low (near the floor / beside the chassis): these are
    # rare under uniform joint sampling, so they are collected separately and over-sampled as targets
    n_low = 0
    while n_low < n // 4:
        # 80% natural postures (shoulder yaw within about +-17 deg, 1 sigma), 20% anything in the joint range
        a, b, c, d = env._sample_arm_config(0.3 if env.rng.random() < 0.8 else None)
        if min(c[0][2], c[1][2]) < 0.30:
            roomy.append(is_roomy())
            lift.append(a); arms.append(b); p.append(c); R.append(d); tg.append(gravity(a, b)); n_low += 1
    # near-home configurations (folded arms) with their TCP poses
    hl, ha, hp, hR, htg = [], [], [], [], []
    d = env.scratch
    for _ in range(n // 4):
        a = env.rng.uniform(0.0, env.lift_hi)
        b = np.clip(env.rng.normal(0.0, 0.05, 12), env.arm_lo, env.arm_hi)
        b[[1, 2, 7, 8]] = env.rng.uniform(0.05, 0.3, 4)      # slightly unfolded, clear of the neighbouring links
        d.qpos[:] = 0.0
        d.qpos[env.q_lift] = a
        d.qpos[env.q_arm] = b
        d.qpos[env.q_fing] = FINGER_OPEN
        mujoco.mj_kinematics(env.m, d)
        hl.append(a); ha.append(b)
        hp.append([d.site_xpos[s].copy() for s in env.site])
        hR.append([d.site_xmat[s].reshape(3, 3).copy() for s in env.site])
        htg.append(gravity(a, b))
    np.savez(POOL, lift=np.array(lift), arms=np.array(arms), p=np.array(p), R=np.array(R),
             h_lift=np.array(hl), h_arms=np.array(ha), h_p=np.array(hp), h_R=np.array(hR),
             tau_g=np.array(tg), h_tau_g=np.array(htg), n_uniform=np.array(n), roomy=np.array(roomy))


def quat2mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(yaw.shape + (3, 3))
    R[..., 0, 0] = c; R[..., 0, 1] = -s; R[..., 1, 0] = s; R[..., 1, 1] = c; R[..., 2, 2] = 1.0
    return R


class Lift3Vec:
    def __init__(self, N, nthread=14, seed=0, max_base_dist=1.5):
        if not os.path.exists(POOL):
            build_pool()
        pool = np.load(POOL)
        self.pool = {k: pool[k] for k in pool.files}
        self.N = N
        self.rng = np.random.default_rng(seed)
        m = self.m = mujoco.MjModel.from_xml_path(XML)
        self.datas = [mujoco.MjData(m) for _ in range(nthread)]
        self.ro = rollout.Rollout(nthread=nthread)
        self.nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        assert self.nstate == 1 + m.nq + m.nv
        jq = lambda n: m.joint(n).qposadr[0]
        arm_names = [f"left_joint1{i}" for i in range(1, 7)] + [f"right_joint2{i}" for i in range(1, 7)]
        fing = ("left_joint17", "left_joint18", "right_joint27", "right_joint28")
        self.q_lift = jq("joint5")
        self.q_arm = np.array([jq(n) for n in arm_names])
        self.q_fing = np.array([jq(n) for n in fing])
        self.arm_lo = np.array([m.joint(n).range[0] for n in arm_names])
        self.arm_hi = np.array([m.joint(n).range[1] for n in arm_names])
        self.lift_hi = float(m.joint("joint5").range[1])
        self.u_lift = m.actuator("joint5").id
        self.u_arm = np.array([m.actuator(n).id for n in arm_names])
        self.u_fing = np.array([m.actuator(n).id for n in fing])
        self.u_wheel = np.array([m.actuator(f"wheel{k}").id for k in range(1, 5)])
        sa = lambda n: m.sensor(n).adr[0]
        self.s_arm = np.array([sa("frc_" + n) for n in arm_names])
        self.s_wheel = np.array([sa(f"frc_wheel{k}") for k in range(1, 5)])
        self.s_touch = np.array([m.sensor(i).adr[0] for i in range(m.nsensor) if m.sensor(i).name.startswith("touch_")])
        self.s_pos = [sa("tcp_l_pos"), sa("tcp_r_pos")]
        self.s_quat = [sa("tcp_l_quat"), sa("tcp_r_quat")]
        self.max_base_dist = max_base_dist

        self.state = np.zeros((N, self.nstate))
        self.ctrl = np.zeros((N, 1, m.nu))
        self.t = np.zeros(N, dtype=np.int64)
        self.arm_tgt = np.zeros((N, 12))
        self.lift_tgt = np.zeros(N)
        self.prev_act = np.zeros((N, ACT_DIM))
        self.tgt_pos = np.zeros((N, 2, 3))
        self.tgt_R = np.zeros((N, 2, 3, 3))
        self.tcp_pos = np.zeros((N, 2, 3))
        self.tcp_R = np.zeros((N, 2, 3, 3))
        self.peak = np.zeros(N)
        self.ep_ret = np.zeros(N)
        self.hold = np.zeros(N, dtype=np.int64)
        self.reset(np.arange(N))

    # views into the state array: [time, qpos, qvel]
    @property
    def qpos(self):
        return self.state[:, 1:1 + self.m.nq]

    @property
    def qvel(self):
        return self.state[:, 1 + self.m.nq:]

    def reset(self, idx):
        n, rng, P = len(idx), self.rng, self.pool
        if n == 0:
            return
        use_pool = rng.random(n) < 0.5
        i_p = rng.integers(0, len(P["lift"]), n)
        i_h = rng.integers(0, len(P["h_lift"]), n)
        lift = np.where(use_pool, P["lift"][i_p], P["h_lift"][i_h])
        arms = np.where(use_pool[:, None], P["arms"][i_p], P["h_arms"][i_h])
        p0 = np.where(use_pool[:, None, None], P["p"][i_p], P["h_p"][i_h])
        R0 = np.where(use_pool[:, None, None, None], P["R"][i_p], P["h_R"][i_h])
        yaw0 = rng.uniform(-math.pi, math.pi, n)
        st = np.zeros((n, self.nstate))
        st[:, 1 + 2] = yaw0
        st[:, 1 + self.q_lift] = lift
        st[:, 1 + self.q_arm] = arms
        st[:, 1 + self.q_fing] = FINGER_OPEN
        self.state[idx] = st
        Rz0 = rotz(yaw0)
        self.tcp_pos[idx] = np.einsum("nij,naj->nai", Rz0, p0)
        self.tcp_R[idx] = np.einsum("nij,najk->naik", Rz0, R0)
        self.arm_tgt[idx] = arms
        self.lift_tgt[idx] = lift
        # targets: TCP poses of another feasible configuration, seen from a displaced base pose
        j = rng.integers(0, len(P["lift"]), n)
        dist = self.max_base_dist * np.sqrt(rng.random(n))
        ang = rng.uniform(-math.pi, math.pi, n)
        off = np.stack([dist * np.cos(ang), dist * np.sin(ang), np.zeros(n)], -1)
        Rz = rotz(rng.uniform(-math.pi, math.pi, n))
        self.tgt_pos[idx] = np.einsum("nij,naj->nai", Rz, P["p"][j]) + off[:, None, :]
        self.tgt_R[idx] = np.einsum("nij,najk->naik", Rz, P["R"][j])
        self.t[idx] = 0
        self.prev_act[idx] = 0.0
        self.peak[idx] = 0.0
        self.ep_ret[idx] = 0.0
        self.hold[idx] = 0

    def errors(self):
        dist = np.linalg.norm(self.tgt_pos - self.tcp_pos, axis=-1)                       # (N, 2)
        tr = np.einsum("naij,naij->na", self.tcp_R, self.tgt_R)
        ang = np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
        return dist, ang

    def obs(self):
        q, v = self.qpos, self.qvel
        yaw = q[:, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        Rb = rotz(yaw)                                                                    # base -> world
        pb = np.stack([q[:, 0], q[:, 1], np.zeros(self.N)], -1)
        vb = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)
        p = np.einsum("nji,naj->nai", Rb, self.tcp_pos - pb[:, None])                     # Rb^T (p - pb)
        R = np.einsum("nji,najk->naik", Rb, self.tcp_R)
        tp = np.einsum("nji,naj->nai", Rb, self.tgt_pos - pb[:, None])
        tR = np.einsum("nji,najk->naik", Rb, self.tgt_R)
        e = tp - p
        n = np.linalg.norm(e, axis=-1, keepdims=True)
        e_dir = e / np.maximum(n, 1e-6) * np.minimum(n, 1.0)
        e_loc = np.einsum("naji,naj->nai", R, e)                                          # error in the gripper frame
        e_loc = np.where(n < 0.2, e_loc * 5.0, e_loc / np.maximum(n, 1e-6))
        r6 = lambda M: np.concatenate([M[..., :, 0], M[..., :, 1]], -1)                   # (N, 2, 6)
        per_arm = np.concatenate([p, r6(R), np.clip(tp, -3.0, 3.0), r6(tR), e_dir, e_loc], -1).reshape(self.N, -1)
        return np.concatenate([
            q[:, self.q_arm], v[:, self.q_arm] * 0.2,
            (q[:, self.q_lift] / self.lift_hi)[:, None], (v[:, self.q_lift] * 10.0)[:, None],
            (self.lift_tgt / self.lift_hi)[:, None],
            q[:, self.q_fing[[0, 2]]] / FINGER_OPEN, vb, self.arm_tgt - q[:, self.q_arm],
            per_arm, self.prev_act], -1).astype(np.float32)

    def step(self, act):
        N = self.N
        act = np.clip(act.astype(np.float64), -1.0, 1.0)
        q_arm = self.qpos[:, self.q_arm]
        self.lift_tgt = np.clip(self.lift_tgt + act[:, 4] * LIFT_VMAX * DT_CTRL, 0.0, self.lift_hi)
        tgt = self.arm_tgt + act[:, 5:17] * ARM_VMAX * DT_CTRL
        tgt = np.clip(tgt, q_arm - TARGET_LEAD, q_arm + TARGET_LEAD)
        self.arm_tgt = np.clip(tgt, self.arm_lo, self.arm_hi)
        u = self.ctrl[:, 0]
        u[:, self.u_wheel] = act[:, 0:4] * WHEEL_VMAX
        u[:, self.u_lift] = self.lift_tgt
        u[:, self.u_arm] = self.arm_tgt
        g = (act[:, 17:19] + 1.0) * 0.5 * FINGER_OPEN
        u[:, self.u_fing] = g[:, [0, 0, 1, 1]]
        state, sens = self.ro.rollout(self.m, self.datas, self.state, self.ctrl, nstep=DECIMATION, skip_checks=False)
        self.state = np.ascontiguousarray(state[:, -1])
        last = sens[:, -1]
        for a in range(2):
            self.tcp_pos[:, a] = last[:, self.s_pos[a]:self.s_pos[a] + 3]
            self.tcp_R[:, a] = quat2mat(last[:, self.s_quat[a]:self.s_quat[a] + 4])
        ratio = np.abs(sens[:, :, self.s_arm]).max(1) / ARM_TPEAK                          # (N, 12)
        wt2 = (sens[:, :, self.s_wheel] ** 2).sum((1, 2))
        contact = (sens[:, :, self.s_touch] > 0.0).any((1, 2))
        # ---- reward
        cur_peak = ratio.max(1)
        new_peak = np.maximum(self.peak, cur_peak)
        d_peak = new_peak - self.peak
        self.peak = new_peak
        dist, ang = self.errors()
        near = np.exp(-dist / 0.2)
        r = (0.5 * (1 - np.tanh(dist / 0.5)) + 1.0 * (1 - np.tanh(dist / 0.06))
             + 0.3 * (1 - np.tanh(ang / 1.5)) + 0.7 * near * (1 - np.tanh(ang / 0.3))).sum(1)
        inside = (dist < POS_TOL) & (ang < ROT_TOL)
        want_open = np.where((dist < 0.02) & (ang < 0.15), 0.0, 1.0)
        r -= 0.2 * np.abs((act[:, 17:19] + 1.0) * 0.5 - want_open).sum(1)
        r += 0.5 * inside.sum(1)
        ok = inside.all(1)
        r += 1.0 * ok
        self.hold = np.where(ok, self.hold + 1, 0)
        r -= 1.0 * cur_peak ** 2 + 0.3 * (ratio ** 2).mean(1) + 60.0 * d_peak
        r -= 0.02 * ((act - self.prev_act) ** 2).sum(1)
        r -= 0.002 * wt2 / (DECIMATION * WHEEL_TMAX ** 2)
        r -= 1.0 * contact
        r *= 0.2
        self.prev_act = act
        self.t += 1
        self.ep_ret += r
        done = self.t >= EPISODE_STEPS
        idx = np.nonzero(done)[0]
        infos = None
        if len(idx):
            infos = np.stack([self.ep_ret[idx], dist[idx, 0], ang[idx, 0], dist[idx, 1], ang[idx, 1], self.peak[idx],
                              (self.hold[idx] >= 25).astype(float), cur_peak[idx]], -1)
            self.reset(idx)
        return self.obs(), r.astype(np.float32), done, infos


if __name__ == "__main__":
    env = Lift3Vec(4096)
    o = env.obs()
    assert o.shape == (4096, OBS_DIM), o.shape
    t = time.time()
    for _ in range(50):
        o, r, d, inf = env.step(np.random.uniform(-1, 1, (4096, ACT_DIM)))
    dt = time.time() - t
    print(f"{4096 * 50 / dt:.0f} env steps/s; reward mean {r.mean():.3f}; obs finite {np.isfinite(o).all()}; absmax {np.abs(o).max():.2f}")

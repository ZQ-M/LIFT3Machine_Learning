"""LIFT3 dual-arm mobile reaching environment (MuJoCo, CPU, batched inside one process).

Task: drive the omni base, the lift and both 6-DoF arms so that each gripper's grasp point (TCP)
reaches its own target frame (position + orientation), then close the gripper, while keeping the
peak arm joint torque low.

Action (19, each in [-1, 1]):
    0:4    wheel speed commands            -> x WHEEL_VMAX rad/s (torque-limited velocity loop per wheel)
    4      lift target velocity            -> x LIFT_VMAX m/s
    5:11   left arm joint target velocity  -> x ARM_VMAX rad/s (integrated into PD position targets)
    11:17  right arm joint target velocity
    17     left gripper   (-1 closed .. +1 open)
    18     right gripper
"""
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "lift3.xml")

DT_CTRL = 0.02
DECIMATION = 4
EPISODE_STEPS = 750                       # 15 s (the lift needs 11.4 s for its full travel)

WHEEL_R = 0.075
WHEEL_XY = [(0.1952, 0.1552, 0.7854), (0.1952, -0.1552, -0.7854),
            (-0.1952, -0.1552, -2.3562), (-0.1952, 0.1552, 2.3562)]   # x, y, axle yaw (URDF joint1..4)
WHEEL_VMAX = 20.0                         # rad/s  (URDF)
WHEEL_TMAX = 7.0                          # N*m    (URDF)
WHEEL_KP = 2.0
WHEEL_DAMP = 0.02
LIFT_VMAX = 0.0333                        # m/s    (URDF)
ARM_VMAX = np.array([3.0, 3.0, 3.0, 5.0, 5.0, 5.0] * 2)
ARM_TPEAK = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0] * 2)
FINGER_OPEN = 0.044
TARGET_LEAD = 0.15                        # PD target may lead the joint by at most this (rad)

POS_TOL, ROT_TOL = 0.015, 0.10            # success: 15 mm and 0.1 rad (5.7 deg) on both arms
ACT_DIM = 19
OBS_DIM = 111


def wheel_rows():
    rows = []
    for x, y, yaw in WHEEL_XY:
        ax, ay = math.cos(yaw), math.sin(yaw)       # axle direction in base frame
        tx, ty = -ay, ax                            # rolling direction t = z x a
        rows.append((tx, ty, x * ty - y * tx))
    return rows


ROWS = wheel_rows()


def rot6(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) * 0.5
    return math.acos(min(1.0, max(-1.0, c)))


class Lift3Batch:
    """M independent environments stepped one after another (used inside a worker process)."""

    def __init__(self, M, seed=0, max_base_dist=1.5, eval_mode=False):
        self.M = M
        self.rng = np.random.default_rng(seed)
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.ds = [mujoco.MjData(self.m) for _ in range(M)]
        self.scratch = mujoco.MjData(self.m)
        m = self.m
        jq = lambda n: m.joint(n).qposadr[0]
        self.q_lift = jq("joint5")
        arm_names = [f"left_joint1{i}" for i in range(1, 7)] + [f"right_joint2{i}" for i in range(1, 7)]
        self.q_arm = np.array([jq(n) for n in arm_names])
        self.q_fing = np.array([jq(n) for n in ("left_joint17", "left_joint18", "right_joint27", "right_joint28")])
        self.arm_lo = np.array([m.joint(n).range[0] for n in arm_names])
        self.arm_hi = np.array([m.joint(n).range[1] for n in arm_names])
        self.lift_hi = float(m.joint("joint5").range[1])
        self.u_lift = m.actuator("joint5").id
        self.u_arm = np.array([m.actuator(n).id for n in arm_names])
        self.u_fing = np.array([m.actuator(n).id for n in ("left_joint17", "left_joint18", "right_joint27", "right_joint28")])
        self.site = [m.site("tcp_l").id, m.site("tcp_r").id]
        self.max_base_dist = max_base_dist
        self.eval_mode = eval_mode

        self.t = np.zeros(M, dtype=np.int64)
        self.arm_tgt = np.zeros((M, 12))
        self.lift_tgt = np.zeros(M)
        self.prev_act = np.zeros((M, ACT_DIM))
        self.tgt_pos = np.zeros((M, 2, 3))
        self.tgt_R = np.zeros((M, 2, 3, 3))
        self.peak = np.zeros(M)                    # running episode peak of |tau| / tau_peak
        self.wheel_tau = np.zeros((M, 4))
        self.ep_ret = np.zeros(M)
        self.hold = np.zeros(M)                    # consecutive steps with both arms in tolerance
        for i in range(M):
            self._reset(i)

    # ------------------------------------------------------------------ sampling
    def _sample_arm_config(self, yaw_sigma=None):
        """Random collision-free lift + arm configuration (base at the origin). Returns (lift, arms) and TCP poses."""
        d = self.scratch
        while True:
            lift = self.rng.uniform(0.0, self.lift_hi)
            arms = self.rng.uniform(self.arm_lo, self.arm_hi)
            if yaw_sigma is not None:      # natural postures: shoulder (root) joints near straight ahead
                arms[[0, 6]] = np.clip(self.rng.normal(0.0, yaw_sigma, 2), self.arm_lo[[0, 6]], self.arm_hi[[0, 6]])
            d.qpos[:] = 0.0
            d.qpos[self.q_lift] = lift
            d.qpos[self.q_arm] = arms
            d.qpos[self.q_fing] = FINGER_OPEN
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_collision(self.m, d)
            if d.ncon > 0:
                continue
            p = [d.site_xpos[s].copy() for s in self.site]
            if min(p[0][2], p[1][2]) < 0.06:        # grasp point at least 6 cm above the floor
                continue
            R = [d.site_xmat[s].reshape(3, 3).copy() for s in self.site]
            return lift, arms, p, R

    def _reset(self, i):
        d = self.ds[i]
        rng = self.rng
        mujoco.mj_resetData(self.m, d)
        # start configuration
        if rng.random() < 0.5:
            lift, arms, _, _ = self._sample_arm_config()
        else:
            lift = rng.uniform(0.0, self.lift_hi)
            arms = np.clip(rng.normal(0.0, 0.05, 12), self.arm_lo, self.arm_hi)
        yaw0 = rng.uniform(-math.pi, math.pi)
        d.qpos[2] = yaw0
        d.qpos[self.q_lift] = lift
        d.qpos[self.q_arm] = arms
        d.qpos[self.q_fing] = FINGER_OPEN
        self.arm_tgt[i] = arms
        self.lift_tgt[i] = lift
        d.ctrl[self.u_lift] = lift
        d.ctrl[self.u_arm] = arms
        d.ctrl[self.u_fing] = FINGER_OPEN
        # targets: TCP poses of another random feasible configuration, seen from a displaced base pose
        _, _, p, R = self._sample_arm_config()
        dist = self.max_base_dist * math.sqrt(rng.random())
        ang = rng.uniform(-math.pi, math.pi)
        bx, by = dist * math.cos(ang), dist * math.sin(ang)
        byaw = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(byaw), math.sin(byaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for a in range(2):
            self.tgt_pos[i, a] = Rz @ p[a] + np.array([bx, by, 0.0])
            self.tgt_R[i, a] = Rz @ R[a]
        mujoco.mj_forward(self.m, d)
        self.t[i] = 0
        self.prev_act[i] = 0.0
        self.peak[i] = 0.0
        self.ep_ret[i] = 0.0
        self.hold[i] = 0
        self.wheel_tau[i] = 0.0

    # ------------------------------------------------------------------ observation
    def _errors(self, i):
        d = self.ds[i]
        out = []
        for a in range(2):
            p = d.site_xpos[self.site[a]]
            R = d.site_xmat[self.site[a]].reshape(3, 3)
            out.append((float(np.linalg.norm(self.tgt_pos[i, a] - p)), rot_angle(R, self.tgt_R[i, a])))
        return out

    def _obs(self, i):
        d = self.ds[i]
        yaw = d.qpos[2]
        c, s = math.cos(yaw), math.sin(yaw)
        Rb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])      # base -> world
        pb = np.array([d.qpos[0], d.qpos[1], 0.0])
        vb = np.array([c * d.qvel[0] + s * d.qvel[1], -s * d.qvel[0] + c * d.qvel[1], d.qvel[2]])
        parts = [d.qpos[self.q_arm], d.qvel[self.q_arm] * 0.2,
                 [d.qpos[self.q_lift] / self.lift_hi, d.qvel[self.q_lift] * 10.0, self.lift_tgt[i] / self.lift_hi],
                 d.qpos[self.q_fing[[0, 2]]] / FINGER_OPEN, vb,
                 self.arm_tgt[i] - d.qpos[self.q_arm]]
        for a in range(2):
            p = Rb.T @ (d.site_xpos[self.site[a]] - pb)
            R = Rb.T @ d.site_xmat[self.site[a]].reshape(3, 3)
            tp = Rb.T @ (self.tgt_pos[i, a] - pb)
            tR = Rb.T @ self.tgt_R[i, a]
            e = tp - p
            n = np.linalg.norm(e)
            parts += [p, rot6(R), np.clip(tp, -3.0, 3.0), rot6(tR), e / max(n, 1e-6) * min(n, 1.0), R.T @ e * 5.0 if n < 0.2 else R.T @ e / n]
        parts.append(self.prev_act[i])
        return np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in parts])

    def obs(self):
        return np.stack([self._obs(i) for i in range(self.M)]).astype(np.float32)

    # ------------------------------------------------------------------ step
    def _step_one(self, i, act):
        m, d = self.m, self.ds[i]
        act = np.clip(act, -1.0, 1.0)
        # --- commands
        wcmd = act[0:4] * WHEEL_VMAX
        self.lift_tgt[i] = min(self.lift_hi, max(0.0, self.lift_tgt[i] + act[4] * LIFT_VMAX * DT_CTRL))
        q = d.qpos[self.q_arm]
        tgt = self.arm_tgt[i] + act[5:17] * ARM_VMAX * DT_CTRL
        tgt = np.clip(tgt, q - TARGET_LEAD, q + TARGET_LEAD)
        self.arm_tgt[i] = np.clip(tgt, self.arm_lo, self.arm_hi)
        d.ctrl[self.u_lift] = self.lift_tgt[i]
        d.ctrl[self.u_arm] = self.arm_tgt[i]
        g = (act[17:19] + 1.0) * 0.5 * FINGER_OPEN
        d.ctrl[self.u_fing] = (g[0], g[0], g[1], g[1])
        # --- physics, with the omni-wheel traction model evaluated every substep
        tau_max = np.zeros(12)
        wt2 = 0.0
        contact = False
        for _ in range(DECIMATION):
            yaw = d.qpos[2]
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy, wz = d.qvel[0], d.qvel[1], d.qvel[2]
            vbx, vby = c * vx + s * vy, -s * vx + c * vy
            fx = fy = mz = 0.0
            for k in range(4):
                r0, r1, r2 = ROWS[k]
                w = -(r0 * vbx + r1 * vby + r2 * wz) / WHEEL_R
                tau = WHEEL_KP * (wcmd[k] - w)
                tau = WHEEL_TMAX if tau > WHEEL_TMAX else (-WHEEL_TMAX if tau < -WHEEL_TMAX else tau)
                self.wheel_tau[i, k] = tau
                wt2 += tau * tau
                f = -(tau - WHEEL_DAMP * w) / WHEEL_R
                fx += r0 * f
                fy += r1 * f
                mz += r2 * f
            d.qfrc_applied[0] = c * fx - s * fy
            d.qfrc_applied[1] = s * fx + c * fy
            d.qfrc_applied[2] = mz
            mujoco.mj_step(m, d)
            np.maximum(tau_max, np.abs(d.actuator_force[self.u_arm]), out=tau_max)
            contact = contact or d.ncon > 0
        # --- reward
        ratio = tau_max / ARM_TPEAK
        cur_peak = float(ratio.max())
        new_peak = max(self.peak[i], cur_peak)
        d_peak = new_peak - self.peak[i]
        self.peak[i] = new_peak
        errs = self._errors(i)
        r = 0.0
        ok = True
        for a in range(2):
            dist, ang = errs[a]
            near = math.exp(-dist / 0.2)
            r += 0.5 * (1.0 - math.tanh(dist / 0.5)) + 1.0 * (1.0 - math.tanh(dist / 0.06))
            r += 0.3 * (1.0 - math.tanh(ang / 1.5)) + 0.7 * near * (1.0 - math.tanh(ang / 0.3))
            inside = dist < POS_TOL and ang < ROT_TOL
            ok = ok and inside
            want_open = 0.0 if (dist < 0.02 and ang < 0.15) else 1.0
            r -= 0.2 * abs((act[17 + a] + 1.0) * 0.5 - want_open)
            if inside:
                r += 0.5
        if ok:
            r += 1.0
            self.hold[i] += 1
        else:
            self.hold[i] = 0
        r -= 1.0 * cur_peak * cur_peak                    # current peak torque ratio (shapes the holding posture)
        r -= 0.3 * float(np.mean(ratio * ratio))
        r -= 60.0 * d_peak                                # sums to 60 x the episode's peak torque ratio
        r -= 0.02 * float(np.sum((act - self.prev_act[i]) ** 2))
        r -= 0.002 * wt2 / (DECIMATION * WHEEL_TMAX ** 2)
        if contact:
            r -= 1.0
        r *= 0.2
        self.prev_act[i] = act
        self.t[i] += 1
        self.ep_ret[i] += r
        done = self.t[i] >= EPISODE_STEPS
        info = None
        if done:
            info = (self.ep_ret[i], errs[0][0], errs[0][1], errs[1][0], errs[1][1], self.peak[i],
                    float(self.hold[i] >= 25), cur_peak)
            self._reset(i)
        return r, done, info

    def step(self, actions):
        rew = np.zeros(self.M, dtype=np.float32)
        done = np.zeros(self.M, dtype=np.bool_)
        infos = []
        for i in range(self.M):
            r, dn, info = self._step_one(i, actions[i].astype(np.float64))
            rew[i] = r
            done[i] = dn
            if info is not None:
                infos.append(info)
        return self.obs(), rew, done, infos


if __name__ == "__main__":
    import time
    env = Lift3Batch(8, seed=1)
    o = env.obs()
    print("obs", o.shape)
    assert o.shape[1] == OBS_DIM, o.shape
    t = time.time()
    n = 300
    for _ in range(n):
        o, r, dn, inf = env.step(np.random.uniform(-1, 1, (8, ACT_DIM)))
    dt = time.time() - t
    print(f"{8 * n / dt:.0f} env steps/s per process; reward mean {r.mean():.3f}; obs absmax {np.abs(o).max():.2f}; finite {np.isfinite(o).all()}")

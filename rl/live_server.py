"""Real-time policy server for the web viewer.

Runs ONE simulated robot at real-time speed (50 Hz control) with the trained policy in the loop and
serves the viewer page plus a small JSON API on the same port:

    GET  /api/state      current joint positions, targets, errors, torque ratios
    POST /api/targets    {"targets": [{"p": [x,y,z], "q": [x,y,z,w]}, {...}]}   (left, right)
    POST /api/reset      put the robot back at the origin in its folded pose
    POST /api/lock       {"lock": [left, right]}  hold that arm in its folded rest posture, bypassing the policy
    POST /api/feasible   same body as /api/targets; answers whether one posture can reach both targets

    ~/miniconda3/envs/rlwp/bin/python live_server.py --ckpt runs/v9/model_last.pt --port 43030
    then open http://127.0.0.1:43030/viewer/

The checkpoint file is re-read whenever it changes, so the live robot follows training progress.
"""
import argparse
import json
import os
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from lift3_env import DT_CTRL, FINGER_OPEN, HERE
from dual_ik import DualIK
from lift3_gpu import Lift3GPU, quat2mat
from train_gpu import ActorCritic

ROOT = os.path.dirname(HERE)


class Live:
    def __init__(self, ckpt):
        self.ckpt, self.mtime, self.iter = ckpt, 0.0, 0
        self.lock = threading.Lock()
        self.env = Lift3GPU(1, seed=0)
        self.env.live = True
        self.dev = self.env.dev
        self.ac = ActorCritic().to(self.dev)
        self.names = [self.env.mjm.joint(j).name for j in range(self.env.mjm.njnt)]
        self.pending_targets, self.pending_reset = None, True
        self.lock_arm = [False, False]
        self.ik, self.ik_lock = DualIK(), threading.Lock()
        self.peak = 0.0
        self.rest_action = self.env.hold_action(torch.zeros(1, device=self.dev), torch.zeros(1, 12, device=self.dev))[0]
        self.state = {"ready": False}
        self.load()

    def load(self):
        try:
            mt = os.path.getmtime(self.ckpt)
            if mt == self.mtime:
                return
            ck = torch.load(self.ckpt, weights_only=False, map_location=self.dev)
            self.ac.load_state_dict(ck["model"])
            self.ac.eval()
            self.mtime, self.iter = mt, ck["iter"]
            print("loaded", self.ckpt, "iter", self.iter, flush=True)
        except Exception as e:                                   # file may be mid-write during training
            print("checkpoint not loaded:", e, flush=True)

    def reset(self):
        e = self.env
        e.qpos[:] = 0.0
        e.qpos[:, e.q_fing] = FINGER_OPEN
        e.qvel[:] = 0.0
        e.warm[:] = 0.0
        e.arm_tgt[:] = 0.0
        e.arm_vcmd[:] = 0.0
        e.lift_tgt[:] = 0.0
        hold = e.hold_action(e.lift_tgt, e.arm_tgt)
        e.prev_act[:] = hold
        self.peak = 0.0
        e.step(hold)                                   # refresh sensors / TCP poses

    def loop(self):
        e = self.env
        obs = None
        k = 0
        nxt = time.perf_counter()
        while True:
            with self.lock:
                tg, rs = self.pending_targets, self.pending_reset
                self.pending_targets, self.pending_reset = None, False
            if rs:
                self.reset()
                if tg is None and obs is None:                 # first start: targets = current gripper poses
                    e.tgt_pos[:] = e.tcp_pos
                    e.tgt_R[:] = e.tcp_R
                obs = e.obs()
            if tg is not None:
                for a in range(2):
                    e.tgt_pos[0, a] = torch.tensor(tg[a]["p"], dtype=torch.float32, device=self.dev)
                    x, y, z, w = tg[a]["q"]
                    e.tgt_R[0, a] = quat2mat(torch.tensor([w, x, y, z], dtype=torch.float32, device=self.dev))
                self.peak = 0.0                                # peak torque is counted per target change
                obs = e.obs()
            # A locked arm is taken out of the policy's hands: its joints are commanded straight to the folded
            # rest posture, and the policy is told that arm is already on its target, so the chassis, the lift
            # and the other arm are not disturbed by it.
            locked = list(self.lock_arm)
            for a in range(2):
                if locked[a]:
                    e.tgt_pos[0, a] = e.tcp_pos[0, a]
                    e.tgt_R[0, a] = e.tcp_R[0, a]
            if any(locked):
                obs = e.obs()
            with torch.no_grad():
                act = self.ac.actor(self.ac.norm(obs)).clone()
            for a in range(2):
                if locked[a]:
                    act[0, 5 + 6 * a:11 + 6 * a] = self.rest_action[5 + 6 * a:11 + 6 * a]
                    act[0, 17 + a] = 1.0
            obs, _, _, _ = e.step(act)
            L = e.last
            ratio = L["ratio"][0].cpu().numpy()
            self.peak = max(self.peak, float(ratio.max()))
            snap = {"ready": True, "t": round(k * DT_CTRL, 2), "iter": self.iter, "joints": self.names,
                    "q": [round(float(v), 4) for v in e.qpos[0].cpu().numpy()],
                    "pos_err_mm": [round(float(v) * 1000, 1) for v in L["dist"][0].cpu().numpy()],
                    "rot_err_deg": [round(float(np.degrees(v)), 1) for v in L["ang"][0].cpu().numpy()],
                    "torque_ratio": [round(float(v), 3) for v in ratio], "peak": round(self.peak, 3),
                    "contact": bool(L["contact"][0]), "lock": locked}
            with self.lock:
                self.state = snap
            k += 1
            if k % 100 == 0:
                self.load()
            nxt += DT_CTRL
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()                      # running behind: do not try to catch up


class Handler(SimpleHTTPRequestHandler):
    live = None

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            with self.live.lock:
                st = self.live.state
            return self._json(st)
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        if self.path.startswith("/api/targets") or self.path.startswith("/api/feasible"):
            tg = data.get("targets")
            if not (isinstance(tg, list) and len(tg) == 2 and all(len(t.get("p", [])) == 3 and len(t.get("q", [])) == 4 for t in tg)):
                return self._json({"error": "need two targets with p[3] and q[4]"}, 400)
            if self.path.startswith("/api/feasible"):
                with self.live.ik_lock:
                    return self._json(self.live.ik.solve(tg))
            with self.live.lock:
                self.live.pending_targets = tg
            return self._json({"ok": True})
        if self.path.startswith("/api/lock"):
            lk = data.get("lock")
            if not (isinstance(lk, list) and len(lk) == 2):
                return self._json({"error": "need lock: [left, right]"}, 400)
            self.live.lock_arm = [bool(lk[0]), bool(lk[1])]
            return self._json({"ok": True})
        if self.path.startswith("/api/reset"):
            with self.live.lock:
                self.live.pending_reset = True
            return self._json({"ok": True})
        return self._json({"error": "unknown"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "v9", "model_last.pt"))
    ap.add_argument("--port", type=int, default=43030)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only; 0.0.0.0 = also other devices on the network")
    args = ap.parse_args()
    live = Live(args.ckpt)
    Handler.live = live
    threading.Thread(target=live.loop, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=ROOT))
    print(f"live policy server: http://{args.host}:{args.port}/viewer/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

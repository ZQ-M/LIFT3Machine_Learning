"""PPO training for the LIFT3 reaching task, fully on the GPU (mujoco_warp physics + torch).
Run with the rlwp conda environment:

    ~/miniconda3/envs/rlwp/bin/python train_gpu.py --name v3 --iters 1500
    ~/miniconda3/envs/rlwp/bin/python train_gpu.py --name v3 --iters 3000 --resume runs/v3/model_last.pt
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from lift3_env import ACT_DIM
from lift3_gpu import OBS_DIM, Lift3GPU

HERE = os.path.dirname(os.path.abspath(__file__))


def mlp(i, o, hidden=(512, 256, 128)):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(0.3))))
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    def update_norm(self, x):
        n = x.shape[0]
        bm, bv = x.mean(0), x.var(0)
        tot = self.obs_count + n
        delta = bm - self.obs_mean
        self.obs_var.copy_((self.obs_var * self.obs_count + bv * n + delta ** 2 * self.obs_count * n / tot) / tot)
        self.obs_mean.add_(delta * n / tot)
        self.obs_count.copy_(tot)

    def norm(self, x):
        return torch.clamp((x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10.0, 10.0)

    def dist(self, obs):
        mu = self.actor(self.norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(self.norm(obs)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v13")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--max-base-dist", type=float, default=1.5)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run = os.path.join(HERE, "runs", args.name)
    os.makedirs(run, exist_ok=True)

    gamma, lam, clip, ent_coef, epochs, n_mb, kl_target = 0.995, 0.95, 0.2, 0.001, 5, 4, 0.01
    lr = 3e-4
    dev = torch.device(args.device)
    ac = ActorCritic().to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    it0 = 0
    if args.resume:
        ck = torch.load(args.resume, weights_only=False, map_location=dev)
        ac.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        it0, lr = ck["iter"], ck["lr"]
        resume_alpha = ck.get("alpha")
        print("resumed from", args.resume, "iter", it0)

    venv = Lift3GPU(args.envs, seed=args.seed + it0, max_base_dist=args.max_base_dist, device=args.device)
    N, T = args.envs, args.horizon
    if args.resume and resume_alpha is not None:
        venv.alpha_max = resume_alpha
    obs = venv.obs()
    if not args.resume:
        ac.update_norm(obs)
    log_path = os.path.join(run, "log.csv")
    new_log = not os.path.exists(log_path)
    logf = open(log_path, "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["iter", "steps", "fps", "rew_step", "ep_return", "pos_err_mm", "rot_err_deg", "success",
                      "peak_torque", "hold_torque", "collided", "std", "lr", "episodes"])
    ep_buf = []
    t_start = time.time()
    for it in range(it0, args.iters):
        # guidance weight: full for the first 60% of this run, then fades to 0.15
        frac = (it - it0) / max(1, args.iters - it0)
        venv.guide_w = 1.0 if frac < 0.6 else max(0.15, 1.0 - (frac - 0.6) / 0.3)
        t0 = time.time()
        t_env = 0.0
        b_obs = torch.zeros(T, N, OBS_DIM, device=dev)
        b_act = torch.zeros(T, N, ACT_DIM, device=dev)
        b_logp, b_val, b_rew, b_done = (torch.zeros(T, N, device=dev) for _ in range(4))
        with torch.no_grad():
            for t in range(T):
                dist = ac.dist(obs)
                act = dist.sample()
                val = ac.value(obs)
                b_obs[t], b_act[t], b_val[t] = obs, act, val
                b_logp[t] = dist.log_prob(act).sum(-1)
                te = time.time()
                o, r, dn, infos = venv.step(act)
                t_env += time.time() - te
                rew, done = r, dn
                rew = rew + gamma * val * venv.timeout.float()     # bootstrap on time-outs only, not on collisions
                b_rew[t], b_done[t] = rew, done.float()
                obs = o
                if infos is not None:
                    ep_buf += infos.cpu().tolist()
            last_val = ac.value(obs)
            adv = torch.zeros(T, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_val if t == T - 1 else b_val[t + 1]
                nd = 1.0 - b_done[t]
                delta = b_rew[t] + gamma * nv * nd - b_val[t]
                gae = delta + gamma * lam * nd * gae
                adv[t] = gae
            ret = adv + b_val
        f = lambda x: x.reshape(T * N, *x.shape[2:])
        fo, fa, flp, fadv, fret, fval = f(b_obs), f(b_act), f(b_logp), f(adv), f(ret), f(b_val)
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        for _ in range(epochs):
            perm = torch.randperm(T * N, device=dev)
            for mb in perm.chunk(n_mb):
                dist = ac.dist(fo[mb])
                logp = dist.log_prob(fa[mb]).sum(-1)
                ratio = (logp - flp[mb]).exp()
                with torch.no_grad():
                    kl = (flp[mb] - logp).mean().item()
                    if kl > 2 * kl_target:
                        lr = max(1e-5, lr / 1.5)
                    elif kl < kl_target / 2:
                        lr = min(1e-3, lr * 1.5)
                    for g in opt.param_groups:
                        g["lr"] = lr
                pl = -torch.min(ratio * fadv[mb], ratio.clamp(1 - clip, 1 + clip) * fadv[mb]).mean()
                v = ac.value(fo[mb])
                vc = fval[mb] + (v - fval[mb]).clamp(-clip, clip)
                vl = torch.max((v - fret[mb]) ** 2, (vc - fret[mb]) ** 2).mean()
                loss = pl + 1.0 * vl - ent_coef * dist.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
        with torch.no_grad():
            ac.log_std.clamp_(np.log(0.05), np.log(1.0))
            ac.update_norm(fo)      # after the update, so rollout and update saw the same statistics
        t_all = time.time() - t0
        fps = T * N / (time.time() - t0)
        if len(ep_buf) >= 200 or it == args.iters - 1:
            e = np.array(ep_buf[-6000:]) if ep_buf else np.zeros((1, 9))
            row = [it + 1, (it + 1) * T * N, int(fps), float(b_rew.mean().item()), e[:, 0].mean(),
                   1000 * (e[:, 1].mean() + e[:, 3].mean()) / 2, np.degrees(e[:, 2].mean() + e[:, 4].mean()) / 2,
                   e[:, 6].mean(), e[:, 5].mean(), e[:, 7].mean(), e[:, 8].mean(), float(ac.log_std.exp().mean().item()), lr, len(e)]
            log.writerow([f"{x:.5g}" if isinstance(x, float) else x for x in row])
            logf.flush()
            if (it + 1) % 10 == 0:
                print(f"it {it + 1:5d} | {fps:6.0f} fps | rew/step {row[3]:.3f} | pos {row[5]:6.1f} mm | rot {row[6]:5.1f} deg | "
                      f"success {row[7]:.2f} | peak tq {row[8]:.2f} | hold tq {row[9]:.2f} | collided {row[10]:.2f} | std {row[11]:.2f} | "
                      f"alpha {venv.alpha_max:.2f} | guide {venv.guide_w:.2f} | env {t_env / t_all:.0%} | {(time.time() - t_start) / 60:.0f} min", flush=True)
            ep_buf = ep_buf[-6000:]
        if (it + 1) % 25 == 0 or it == args.iters - 1:
            ck = {"model": ac.state_dict(), "opt": opt.state_dict(), "iter": it + 1, "lr": lr, "alpha": venv.alpha_max}
            torch.save(ck, os.path.join(run, "model_last.pt"))
            if (it + 1) % 100 == 0:
                torch.save(ck, os.path.join(run, f"model_{it + 1}.pt"))


if __name__ == "__main__":
    main()

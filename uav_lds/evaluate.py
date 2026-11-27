"""
Evaluate UAV-LDS checkpoints (Table V protocol): N episodes (default 100), episode e uses seed base_seed + e
(Unity environment parameter "seed" and python / numpy / torch), 1000 steps, parcel limits 100 / 100,
Unity engine at quality 5 with a fixed 30 fps frame clock, no graphics.
AS (number of successes) = small deliveries + 2 x big deliveries (a big parcel is carried by two UAVs).

  python evaluate.py --model all                                     # the 20 pretrained models
  python evaluate.py --model MF-MAAC-P --checkpoint path/to/best_model.pt
  python evaluate.py --model MF-MAAC-P --run_dir results/train/MF-MAAC-P_s41   # best_model.pt and final_model.pt

Policies: MAAC family and MAPPO stochastic (sampled actions), value-decomposition models greedy (--mode overrides).
Output: <out>/<model>__<tag>__<mode>.json with per-episode rows and a summary (read by summarize.py).
Note: Unity physics is not bit-reproducible, so repeated evaluations of one checkpoint differ slightly.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import models as M

sys.path.insert(0, str(M.LIB))
from uav_eval_env import UAVEvalEnv  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Box:                                   # MAPPO utils identify spaces by class name
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def make_actor(model, path, nag, od, na, mode):
    """-> (reset_fn, act_fn(obs_list) -> int actions (nag,))"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    k = M.kind(model)
    if k == 'maac':
        sys.path.insert(0, str(M.LIB / 'MAAC'))
        from utils.agents import AttentionAgent
        ip = ck['init_dict']
        agents = []
        for p, params in zip(ip['agent_init_params'], ck['agent_params']):
            assert p['num_in_pol'] == od and p['num_out_pol'] == na, (p, od, na)
            a = AttentionAgent(p['num_in_pol'], p['num_out_pol'], hidden_dim=ip['pol_hidden_dim'])
            a.load_params(params)
            a.policy.eval()
            agents.append(a)
        explore = mode == 'stochastic'

        def act(obs):
            with torch.no_grad():
                acts = [ag.step(torch.tensor(o, dtype=torch.float32).unsqueeze(0), explore=explore)
                        for ag, o in zip(agents, obs)]
            return np.array([a.numpy()[0].argmax() for a in acts])
        return (lambda: None), act

    if k == 'pymarl':
        import torch.nn as nn
        if mode != 'deterministic':
            raise SystemExit('%s: value-decomposition models are evaluated greedily (--mode deterministic)' % model)
        sys.path.insert(0, str(M.LIB / M.PYMARL_DIR[model]))
        sd = ck['agent']
        h, fin = sd['fc1.weight'].shape
        args = SimpleNamespace(n_actions=na, rnn_hidden_dim=h, n_agents=nag)
        if model == 'MF-QMIX':
            from modules.agents.mf_rnn_agent import MeanFieldRNNAgent
            net = MeanFieldRNNAgent(od, args)
        else:
            from modules.agents.rnn_agent import RNNAgent
            net = RNNAgent(od, args)
            if fin == od:                    # checkpoints without the agent-id one-hot in the agent input
                net.fc1 = nn.Linear(od, h)
                net._append_agent_id = lambda x: x
        net.load_state_dict(sd)
        net.eval()
        st = {}

        def reset():
            st['h'] = net.init_hidden().unsqueeze(0).expand(1, nag, -1)
            st['last'] = np.zeros((nag, na), dtype=np.float32)

        def act(obs):
            o = torch.tensor(np.array(obs), dtype=torch.float32)
            with torch.no_grad():
                if model == 'MF-QMIX':
                    o1, la = o.unsqueeze(0), torch.tensor(st['last']).unsqueeze(0)
                    mo = (o1.sum(1, keepdim=True) - o1) / (nag - 1)
                    ma = (la.sum(1, keepdim=True) - la) / (nag - 1)
                    q, st['h'] = net(o1, la, mo, ma, torch.eye(nag).unsqueeze(0), st['h'])
                else:
                    q, st['h'] = net(o, st['h'])
            a = q.reshape(nag, na).argmax(-1).numpy()          # greedy (all actions available)
            st['last'] = np.eye(na, dtype=np.float32)[a]
            return a
        return reset, act

    # MAPPO
    sys.path.insert(0, str(M.LIB / 'MAPPO'))
    from algorithms.r_actor_critic import R_Actor
    args = SimpleNamespace(hidden_size=128, layer_N=2, use_ReLU=True, use_feature_normalization=True,
                           use_orthogonal=True, gain=0.01, stacked_frames=1,
                           use_policy_active_masks=False, algorithm_name='rmappo')
    actor = R_Actor(args, Box((od,)), Discrete(na), device=torch.device('cpu'))
    actor.load_state_dict(ck['actor'])
    actor.eval()

    def reset():
        pass

    def act(obs):
        with torch.no_grad():
            a, _ = actor(np.array(obs, dtype=np.float32), deterministic=(mode == 'deterministic'))
        return a.cpu().numpy().squeeze(-1).astype(np.int64)
    return reset, act


def evaluate(env, model, path, mode, n, base_seed, steps):
    nag, od, na = env.nagent, env.obs_dim, env.n_branches
    reset_fn, act = make_actor(model, path, nag, od, na, mode)
    rows = []
    for ep in range(n):
        seed = base_seed + ep
        set_seed(seed)
        obs = env.reset(seed=seed)
        reset_fn()
        rew = 0.0
        info = {}
        for _ in range(steps):
            ai = act(obs)
            obs, r, _, info = env.step(np.eye(na, dtype=np.int32)[ai])
            rew += float(np.sum(r))
        rows.append({'seed': seed, 'small': info['small_delivered'], 'big': info['big_delivered'],
                     'AS': info['small_delivered'] + 2 * info['big_delivered'], 'reward': rew})
        print('  episode %3d seed %3d | small %2d big %2d AS %2d | reward %.1f'
              % (ep + 1, seed, rows[-1]['small'], rows[-1]['big'], rows[-1]['AS'], rew), flush=True)
    return rows


def jobs(a):
    """-> list of (model, checkpoint path, tag, seed)"""
    names = M.MODELS if a.model == ['all'] else a.model
    for m in names:
        if m not in M.MODELS:
            raise SystemExit('unknown model %s; choose from %s' % (m, ', '.join(M.MODELS)))
    if a.checkpoint:
        if len(names) != 1:
            raise SystemExit('--checkpoint takes a single --model')
        return [(names[0], Path(a.checkpoint), Path(a.checkpoint).stem, a.seed)]
    if a.run_dir:
        if len(names) != 1:
            raise SystemExit('--run_dir takes a single --model')
        d = Path(a.run_dir)
        cfg = json.load(open(d / 'config.json')) if (d / 'config.json').exists() else {}
        seed = a.seed if a.seed is not None else cfg.get('seed')
        return [(names[0], d / f, tag, seed) for f, tag in (('best_model.pt', 'best'), ('final_model.pt', 'final'))
                if (d / f).exists()]
    return [(m, M.PRETRAINED / ('%s.pt' % m), 'pretrained', None) for m in names]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', nargs='+', required=True, help='model names (see models.py) or "all"')
    p.add_argument('--checkpoint', help='checkpoint file (default: ../pretrained/uav_lds/<model>.pt)')
    p.add_argument('--run_dir', help='training run directory of train.py: evaluates best_model.pt and final_model.pt')
    p.add_argument('--seed', type=int, default=None, help='training seed recorded in the output (summarize.py)')
    p.add_argument('--mode', choices=['stochastic', 'deterministic'], help='default: the paper setting per model')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--base_seed', type=int, default=1)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--worker', type=int, default=50, help='Unity worker id (port offset), unique per process')
    p.add_argument('--build', default=None, help='Unity executable (default: unity_build/Build_Linux_eval_v2_5uav/'
                                                 'Logistics.x86_64 or $UAVLDS_BUILD)')
    p.add_argument('--out', default='results/eval')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--tag', help=argparse.SUPPRESS)      # output name tag, set when called per checkpoint
    a = p.parse_args()
    js = jobs(a)
    if len(js) > 1:
        # one process per checkpoint: the algorithm folders share package names (utils, algorithms, modules)
        for model, path, tag, seed in js:
            cmd = [sys.executable, __file__, '--model', model, '--checkpoint', str(path), '--tag', tag,
                   '--episodes', str(a.episodes), '--base_seed', str(a.base_seed), '--steps', str(a.steps),
                   '--worker', str(a.worker), '--out', a.out]
            cmd += ['--seed', str(seed)] if seed is not None else []
            cmd += ['--mode', a.mode] if a.mode else []
            cmd += ['--build', a.build] if a.build else []
            cmd += ['--overwrite'] if a.overwrite else []
            r = subprocess.run(cmd)
            if r.returncode:
                sys.exit(r.returncode)
        return
    todo = []
    for model, path, tag, seed in js:
        tag = a.tag or tag
        mode = a.mode or M.eval_mode(model)
        out = Path(a.out) / ('%s__%s%s__%s.json' % (model, tag, '' if seed is None else '_s%d' % seed, mode))
        if out.exists() and not a.overwrite:
            print('exists', out)
            continue
        if not path.exists():
            raise SystemExit('checkpoint not found: %s' % path)
        todo.append((model, path, tag, seed, mode, out))
    if not todo:
        return
    build = M.build_path(a.build)
    if not build.exists():
        raise SystemExit('Unity build not found: %s (see README, "UAV-LDS environment")' % build)
    torch.set_num_threads(1)
    os.environ.pop('DISPLAY', None)
    env = UAVEvalEnv(str(build), worker_id=a.worker, quality_level=5, target_frame_rate=30, capture_frame_rate=30,
                     no_graphics=True, width=400, height=200)
    try:
        for model, path, tag, seed, mode, out in todo:
            print('[eval] %s %s (%s)' % (model, path, mode), flush=True)
            t0 = time.time()
            rows = evaluate(env, model, path, mode, a.episodes, a.base_seed, a.steps)
            arr = lambda k: np.array([r[k] for r in rows], dtype=float)
            summary = dict(model=model, checkpoint=str(path), tag=tag, train_seed=seed, build=str(build), mode=mode,
                           n_episodes=a.episodes, base_seed=a.base_seed, steps=a.steps,
                           AS_mean=arr('AS').mean(), AS_std=arr('AS').std(),
                           small_mean=arr('small').mean(), big_mean=arr('big').mean(),
                           reward_mean=arr('reward').mean(), seconds=round(time.time() - t0, 1))
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w') as f:
                json.dump({'summary': summary, 'episodes': rows}, f, indent=2)
            print('%-13s %-10s %-13s AS %6.2f +- %5.2f  (small %.2f, big %.2f)  %.0fs'
                  % (model, tag, mode, summary['AS_mean'], summary['AS_std'], summary['small_mean'],
                     summary['big_mean'], summary['seconds']), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()

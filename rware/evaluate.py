"""
Evaluate RWARE checkpoints (Table V protocol): N episodes (default 100), episode e uses seed base_seed + e
(python / numpy / torch and the environment are re-seeded before each reset), 500 steps.
AS (number of successes) = target-shelf deliveries + shelf returns per episode.

  python evaluate.py --model all                                     # the 20 pretrained models
  python evaluate.py --model MF-MAAC-P --checkpoint path/to/best_model.pt
  python evaluate.py --model MF-MAAC-P --run_dir results/train/MF-MAAC-P_s41   # best_model.pt and model.pt

Policies: MAAC family and MAPPO stochastic (sampled actions), value-decomposition models greedy (--mode overrides).
Output: <out>/<model>__<tag>__<mode>.json with per-episode rows and a summary (read by summarize.py).
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

sys.path[:0] = [str(M.LIB / 'MAAC' / 'baselines-master'), str(M.LIB / 'robotic-warehouse')]
os.environ['PYTHONPATH'] = M.python_path()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_maac(path, explore, n, base_seed):
    sys.path.insert(0, str(M.LIB / 'MAAC'))
    import gym
    import rware  # noqa: F401  (registers the environments)
    from utils.agents import AttentionAgent
    ck = torch.load(path, map_location='cpu', weights_only=False)
    env = gym.make(M.ENV_ID, render_mode='rgb_array', reward_mode='custom', reward_alpha=1.0, reward_beta=0.0)
    ip = ck['init_dict']
    agents = []
    for p, params in zip(ip['agent_init_params'], ck['agent_params']):
        a = AttentionAgent(p['num_in_pol'], p['num_out_pol'], hidden_dim=ip['pol_hidden_dim'])
        a.load_params(params)
        a.policy.eval()
        agents.append(a)
    rows = []
    for ep in range(n):
        seed = base_seed + ep
        set_seed(seed)
        env.seed(seed)
        obs = env.reset()
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            obs = obs[0]
        obs = np.stack([np.asarray(o, dtype=np.float32) for o in obs])
        rew = 0.0
        dl = rt = 0
        for _ in range(500):
            with torch.no_grad():
                acts = [ag.step(torch.from_numpy(obs[i:i + 1]), explore=explore) for i, ag in enumerate(agents)]
            ia = [int(a.numpy()[0].argmax()) for a in acts]
            res = env.step(ia)
            if len(res) == 5:
                obs, r, dones, _, info = res
            else:
                obs, r, dones, info = res
            obs = np.stack([np.asarray(o, dtype=np.float32) for o in obs])
            rew += float(np.sum(r))
            dl += info.get('step_delivery', 0)
            rt += info.get('step_return', 0)
            if any(dones):
                break
        rows.append({'seed': seed, 'delivered': dl, 'returned': rt, 'AS': dl + rt, 'reward': rew})
    env.close()
    return rows


def run_pymarl(model, path, n, base_seed):
    import torch.nn as nn
    sys.path.insert(0, str(M.LIB))
    import train_pymarl_baseline as B
    algo = M.PYMARL[model][0]
    sys.path.insert(0, str(B.ALGO_DIRS[algo]))
    from rware_wrapper import RWARE_QPLEX_Wrapper
    if algo == 'qplex':
        from algorithms.qplex_agent import QPLEXAgent as AgentCls
    elif algo == 'mfqmix':
        from algorithms.mf_qmix_agent import MFQMIXAgent as AgentCls
    elif algo == 'qmix':
        from algorithms.qmix_pymarl_agent import QMIXPyMARLAgent as AgentCls
    else:
        from algorithms.qtran_agent import QTRANAgent as AgentCls
    cfg = dict(B.ALGO_CONFIG[algo])
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['agent']
    h, fin = sd['fc1.weight'].shape
    cfg.update({'algo': algo, 'device': 'cpu', 'rnn_hidden_dim': h, 'env_id': M.ENV_ID})
    env = RWARE_QPLEX_Wrapper(env_id=cfg['env_id'], seed=base_seed, render_mode='rgb_array')
    nag, od, na = env.nagent, env.observation_space[0][0], env.action_space_n[0]
    cfg['unit_dim'] = od
    if algo == 'qplex':
        cfg.update(num_kernel=4, adv_hypernet_layers=1, adv_hypernet_embed=cfg['hypernet_embed'])
    agent = AgentCls(cfg, od, na, od * nag, nag)
    if algo in ('qplex', 'qtran') and fin == od:      # these runs have no agent-id one-hot in the agent input
        agent.agent.fc1 = nn.Linear(od, h)
        agent.agent._append_agent_id = lambda x: x
    agent.agent.load_state_dict(sd)
    agent.agent.eval()
    actor = B.Actor(algo if algo in ('mfqmix', 'qmix') else 'qplex', agent, nag, na)
    rows = []
    for ep in range(n):
        seed = base_seed + ep
        r, d, t = B.play_greedy(env, actor, 500, seed=seed)
        rows.append({'seed': seed, 'delivered': d, 'returned': t, 'AS': d + t, 'reward': r})
    return rows


def run_mappo(path, deterministic, n, base_seed):
    sys.path.insert(0, str(M.LIB))
    import train_mappo as T                 # puts MAPPO/ on sys.path
    from algorithms.rMAPPOPolicy import R_MAPPOPolicy
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(T.CONFIG, use_gpu=False)
    args = SimpleNamespace(**cfg)
    env = T.RWARE_MAPPO_Wrapper(M.ENV_ID, reward_beta=None)
    nag, od, na = env.nagent, env.observation_space[0][0], env.action_space[0]

    class Box:
        def __init__(self, shape):
            self.shape = shape

    class Discrete:
        def __init__(self, k):
            self.n = k
    policy = R_MAPPOPolicy(args, Box((od,)), Box((od * nag,)), Discrete(na), device=torch.device('cpu'))
    policy.actor.load_state_dict(ck['actor'])
    policy.actor.eval()
    rows = []
    for ep in range(n):
        seed = base_seed + ep
        T.set_up_seed(seed)
        env.env.seed(seed)
        obs = np.array(env.reset(), dtype=np.float32)
        rew = dl = rt = 0.0
        for _ in range(500):
            with torch.no_grad():
                act = policy.act(obs, deterministic=deterministic)
            obs, r, dones, info = env.step(act.cpu().numpy())
            obs = np.array(obs, dtype=np.float32)
            rew += float(sum(r))
            dl += info.get('step_delivery', 0)
            rt += info.get('step_return', 0)
            if all(dones):
                break
        rows.append({'seed': seed, 'delivered': dl, 'returned': rt, 'AS': dl + rt, 'reward': rew})
    env.close()
    return rows


def evaluate(model, path, mode, n, base_seed):
    k = M.kind(model)
    if k == 'maac':
        return run_maac(path, mode == 'stochastic', n, base_seed)
    if k == 'pymarl':
        if mode != 'deterministic':
            raise SystemExit('%s: value-decomposition models are evaluated greedily (--mode deterministic)' % model)
        return run_pymarl(model, path, n, base_seed)
    return run_mappo(path, mode == 'deterministic', n, base_seed)


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
        return [(names[0], d / f, tag, seed) for f, tag in (('best_model.pt', 'best'), ('model.pt', 'final'))
                if (d / f).exists()]
    return [(m, M.PRETRAINED / ('%s.pt' % m), 'pretrained', None) for m in names]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', nargs='+', required=True, help='model names (see models.py) or "all"')
    p.add_argument('--checkpoint', help='checkpoint file (default: ../pretrained/rware/<model>.pt)')
    p.add_argument('--run_dir', help='training run directory of train.py: evaluates best_model.pt and model.pt')
    p.add_argument('--seed', type=int, default=None, help='training seed recorded in the output (summarize.py)')
    p.add_argument('--mode', choices=['stochastic', 'deterministic'], help='default: the paper setting per model')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--base_seed', type=int, default=1)
    p.add_argument('--out', default='results/eval')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--tag', help=argparse.SUPPRESS)      # output name tag, set when called per checkpoint
    a = p.parse_args()
    torch.set_num_threads(1)
    todo = jobs(a)
    if len(todo) > 1:
        # one process per checkpoint: the algorithm folders share package names (utils, algorithms, modules)
        for model, path, tag, seed in todo:
            cmd = [sys.executable, __file__, '--model', model, '--checkpoint', str(path), '--episodes', str(a.episodes),
                   '--base_seed', str(a.base_seed), '--out', a.out]
            cmd += ['--seed', str(seed)] if seed is not None else []
            cmd += ['--mode', a.mode] if a.mode else []
            cmd += ['--overwrite'] if a.overwrite else []
            r = subprocess.run(cmd + ['--tag', tag])
            if r.returncode:
                sys.exit(r.returncode)
        return
    for model, path, tag, seed in todo:
        tag = a.tag or tag
        mode = a.mode or M.eval_mode(model)
        out = Path(a.out) / ('%s__%s%s__%s.json' % (model, tag, '' if seed is None else '_s%d' % seed, mode))
        if out.exists() and not a.overwrite:
            print('exists', out)
            continue
        if not path.exists():
            raise SystemExit('checkpoint not found: %s' % path)
        t0 = time.time()
        rows = evaluate(model, path, mode, a.episodes, a.base_seed)
        arr = lambda k: np.array([r[k] for r in rows], dtype=float)
        summary = dict(model=model, checkpoint=str(path), tag=tag, train_seed=seed, env_id=M.ENV_ID, mode=mode,
                       n_episodes=a.episodes, base_seed=a.base_seed,
                       AS_mean=arr('AS').mean(), AS_std=arr('AS').std(),
                       delivered_mean=arr('delivered').mean(), returned_mean=arr('returned').mean(),
                       reward_mean=arr('reward').mean(), seconds=round(time.time() - t0, 1))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'summary': summary, 'episodes': rows}, f, indent=2)
        print('%-13s %-10s %-13s AS %6.2f +- %5.2f  (deliveries %.2f, returns %.2f)  %.0fs'
              % (model, tag, mode, summary['AS_mean'], summary['AS_std'], summary['delivered_mean'],
                 summary['returned_mean'], summary['seconds']), flush=True)


if __name__ == '__main__':
    main()

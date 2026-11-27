"""
Unified runner for the PyMARL-style baselines (QPLEX, MF-QMIX) on RWARE.

The training loop is the one from QPLEX/main_qplex.py and QMIX/main_qmix.py
(same buffer, update-per-episode, target update interval, epsilon schedule,
team reward = sum of agent rewards, terminated=False at the time limit).
Algorithm code in QPLEX/ and QMIX/ is not modified. What this runner adds:
CLI config, the reward_mode switch, per-episode CSV metrics (SR etc.),
best-by-SR checkpoint, and a final 50-episode greedy evaluation, all written
into one results directory.

Example:
  python train_pymarl_baseline.py --algo qplex --reward_mode original --seed 1 \
      --n_episodes 3000 --run_dir results/2026-10-13_QPLEX_original_s1
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ALGO_DIRS = {'qplex': HERE / 'QPLEX', 'mfqmix': HERE / 'QMIX', 'qmix': HERE / 'QMIX',
             'qtran': HERE / 'QTRAN'}   # QTRAN: training loop of the original main_qtran.py
ENV_IDS = {'custom': 'rware-small-5ag-twostage-v2',
           'event': 'rware-small-5ag-twostage-event-v2',
           'original': 'rware-small-5ag-twostage-orig-v2'}

# configs copied from QPLEX/main_qplex.py and QMIX/main_qmix.py (__main__)
COMMON = {
    'episode_limit': 500, 'render_mode': 'rgb_array',
    'buffer_size': 2000, 'batch_size': 32, 'burn_in_period': 32,
    'lr': 0.0005, 'optim_alpha': 0.99, 'optim_eps': 0.00001, 'gamma': 0.99,
    'grad_norm_clip': 10, 'target_update_interval': 200, 'double_q': True,
    'action_selector': 'epsilon_greedy', 'epsilon_start': 1.0, 'epsilon_finish': 0.05,
    # network sizes raised from the original mains (64/32/64) to match MF-MAAC-P's 128 hidden
    'rnn_hidden_dim': 128, 'mixing_embed_dim': 64, 'hypernet_embed': 128,
}
ALGO_CONFIG = {
    # epsilon_anneal_time: per-algorithm best value from earlier RWARE runs
    # (QPLEX 50K, MF-QMIX 800K; candidates {50K, 500K, 800K}). Override: --epsilon_anneal_time
    # The 2026-10-13 3,000-episode tests T2/T4 used 500K for both (recorded in their config.json).
    'qplex': dict(COMMON, model_name='QPLEX_RWARE', epsilon_anneal_time=50000,
                  n_head=4, attend_reg_coef=0.001, nonlinear=False, weighted_head=True,
                  mask_dead=False, state_bias=True, is_minus_one=True),
    'mfqmix': dict(COMMON, model_name='MF-QMIX_RWARE', epsilon_anneal_time=800000,
                   mixer='qmix', n_moments=1),
    # 2026-10-22: QMIX, authors' official PyMARL setting (QMIX/algorithms/qmix_pymarl_agent.py):
    # agent input obs + last action + agent id, QMixer with 2-layer hypernetworks on the global state
    'qmix': dict(COMMON, model_name='QMIX_RWARE', epsilon_anneal_time=800000, mixer='qmix'),
    # config of the paper QTRAN run
    'qtran': dict(COMMON, model_name='QTRAN_RWARE', epsilon_anneal_time=1000000, rnn_hidden_dim=128,
                  mixing_embed_dim=64, qtran_arch='qtran_paper', network_size='small', opt_loss=1.0,
                  nopt_min_loss=0.1),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Actor(object):
    """Hides the step() signature difference (MF-QMIX needs the last joint action)."""
    def __init__(self, algo, agent, n_agents, n_actions):
        self.algo, self.agent = algo, agent
        self.eye = np.eye(n_actions, dtype=np.float32)
        self.n_agents, self.n_actions = n_agents, n_actions
        self.last = None

    def reset(self):
        self.agent.init_hidden(batch_size=1)
        self.last = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)

    def act(self, obs, avail, t_env, test_mode):
        with torch.no_grad():
            if self.algo in ('mfqmix', 'qmix'):      # both take the last joint action
                a = self.agent.step(obs, self.last, avail, t_env, test_mode=test_mode)
            else:
                a = self.agent.step(obs, avail, t_env, test_mode=test_mode)
        self.last = self.eye[a]
        return a


def play_greedy(env, actor, limit, seed=None):
    if seed is not None:
        set_seed(seed)
        env.env.seed(seed)
    obs = env.reset()
    actor.reset()
    rew = dl = rt = 0.0
    for _ in range(limit):
        a = actor.act(obs, env.get_avail_actions(), 0, test_mode=True)
        obs, r, dones, info = env.step(a)
        rew += float(sum(r))
        dl += info.get('step_delivery', 0)
        rt += info.get('step_return', 0)
        if all(dones):
            break
    return rew, dl, rt


def evaluate_ckpt(path, env, agent, actor, cfg, n_episodes, base_seed=10000):
    ck = torch.load(path, map_location=cfg['device'])
    agent.agent.load_state_dict(ck['agent'])
    rows = []
    for ep in range(n_episodes):
        r, d, t = play_greedy(env, actor, cfg['episode_limit'], seed=base_seed + ep)
        rows.append({'seed': base_seed + ep, 'delivered': d, 'returned': t, 'SR': d + t, 'reward': r})
    a = lambda k: np.array([x[k] for x in rows], dtype=float)
    summary = {'model': Path(path).name, 'algo': cfg['algo'], 'reward_mode': cfg['reward_mode'],
               'seed': cfg['seed'], 'n_episodes': n_episodes, 'base_seed': base_seed,
               'explore': False, 'SR_mean': a('SR').mean(), 'SR_std': a('SR').std(),
               'ATCD': a('delivered').mean(), 'ASR': a('returned').mean(),
               'reward_mean': a('reward').mean()}
    with open(Path(path).with_suffix('.eval.json'), 'w') as f:
        json.dump({'summary': summary, 'episodes': rows}, f, indent=2)
    print('[eval] %s: SR %.2f +- %.2f (ATCD %.2f, ASR %.2f)'
          % (Path(path).name, summary['SR_mean'], summary['SR_std'], summary['ATCD'], summary['ASR']))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', required=True, choices=list(ALGO_DIRS))
    p.add_argument('--reward_mode', default='original', choices=list(ENV_IDS))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--n_episodes', type=int, default=3000)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--epsilon_anneal_time', type=int, default=None)
    p.add_argument('--reward_beta', type=float, default=None,
                   help='team-term weight for custom/event (env default 0.1; old Table IV runs 0.0)')
    p.add_argument('--rnn_hidden_dim', type=int, default=None)
    p.add_argument('--mixing_embed_dim', type=int, default=None)
    p.add_argument('--hypernet_embed', type=int, default=None)
    p.add_argument('--obs_agent_id', default=None, choices=['on', 'off'],
                   help='QPLEX only: off = observation-only agent input (old-project run1); default on')
    p.add_argument('--greedy_eval_every', type=int, default=100)
    p.add_argument('--save_best_sr', type=int, default=1)
    p.add_argument('--wandb_mode', default='disabled')
    p.add_argument('--wandb_project', default='MF-MAAC_RWARE')
    p.add_argument('--wandb_entity', default=None)
    a = p.parse_args()

    sys.path.insert(0, str(ALGO_DIRS[a.algo]))
    from rware_wrapper import RWARE_QPLEX_Wrapper
    from components.episode_buffer import EpisodeBatch, ReplayBuffer
    from components.transforms import OneHot
    if a.algo == 'qplex':
        from algorithms.qplex_agent import QPLEXAgent as AgentCls
    elif a.algo == 'qmix':
        from algorithms.qmix_pymarl_agent import QMIXPyMARLAgent as AgentCls
    elif a.algo == 'qtran':
        from algorithms.qtran_agent import QTRANAgent as AgentCls
    else:
        from algorithms.mf_qmix_agent import MFQMIXAgent as AgentCls

    cfg = dict(ALGO_CONFIG[a.algo])
    cfg.update({'algo': a.algo, 'reward_mode': a.reward_mode, 'env_id': ENV_IDS[a.reward_mode],
                'seed': a.seed, 'n_episodes': a.n_episodes,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    for k in ('epsilon_anneal_time', 'rnn_hidden_dim', 'mixing_embed_dim', 'hypernet_embed'):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.obs_agent_id is not None:
        cfg['obs_agent_id'] = a.obs_agent_id == 'on'
    set_seed(cfg['seed'])
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'config.json').exists():
        raise SystemExit('run_dir already holds a run: %s' % run_dir)
    cfg['run_dir'] = str(run_dir)

    env = RWARE_QPLEX_Wrapper(env_id=cfg['env_id'], seed=cfg['seed'], render_mode=cfg['render_mode'])
    eval_env = RWARE_QPLEX_Wrapper(env_id=cfg['env_id'], seed=cfg['seed'] + 777, render_mode=cfg['render_mode'])
    assert env.env.unwrapped.reward_mode == cfg['reward_mode']
    if a.reward_beta is not None:                      # used by _compute_reward every step
        for e in (env, eval_env):
            e.env.unwrapped.reward_beta = a.reward_beta
    n_agents = env.nagent
    obs_dim = env.observation_space[0][0]
    n_actions = env.action_space_n[0]
    state_dim = obs_dim * n_agents
    cfg['unit_dim'] = obs_dim
    if a.algo == 'qplex':
        cfg['num_kernel'] = 4
        cfg['adv_hypernet_layers'] = 1
        cfg['adv_hypernet_embed'] = cfg['hypernet_embed']

    scheme = {
        "state": {"vshape": (state_dim,)},
        "obs": {"vshape": (obs_dim,), "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    if a.algo == 'qtran':                              # scheme of main_qtran.py
        scheme["actions_onehot"] = {"vshape": (n_actions,), "group": "agents", "dtype": torch.float}
    groups = {"agents": n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=n_actions)])}
    agent = AgentCls(cfg, obs_dim, n_actions, state_dim, n_agents)
    buffer = ReplayBuffer(scheme, groups, cfg['buffer_size'], cfg['episode_limit'] + 1,
                          cfg['burn_in_period'], preprocess=preprocess, device="cpu")
    actor = Actor(a.algo, agent, n_agents, n_actions)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    import wandb
    # human-readable wandb identity: model | reward | budget | epsilon schedule | seed
    model = {'qplex': 'QPLEX', 'mfqmix': 'MF-QMIX', 'qmix': 'QMIX', 'qtran': 'QTRAN'}[a.algo]
    reward = cfg['reward_mode'] + ' reward'
    if cfg['reward_mode'] != 'original':
        cfg['reward_beta'] = float(env.env.unwrapped.reward_beta)
        reward += ' (beta=%g)' % cfg['reward_beta']
    budget = '%gk ep' % (cfg['n_episodes'] / 1000.0)
    ea = cfg['epsilon_anneal_time']
    eps = 'eps-anneal ' + ('%gM' % (ea / 1e6) if ea >= 1e6 else '%gK' % (ea / 1e3))
    hid = 'rnn%d/mix%d' % (cfg['rnn_hidden_dim'], cfg['mixing_embed_dim'])
    cfg['model'], cfg['reward'] = model, reward
    sys.path.insert(0, str(HERE))
    from wandb_naming import run_name
    wb_name, cfg['model_version'] = run_name(model, cfg['reward_mode'], consume=a.wandb_mode == 'online')
    wandb.init(entity=a.wandb_entity, project=a.wandb_project, name=wb_name,
               group=model, tags=[model, cfg['reward_mode']],
               config=cfg, mode=a.wandb_mode, dir=str(run_dir))
    with open(run_dir / 'config.json', 'w') as f:      # re-dump with model/reward labels
        json.dump(cfg, f, indent=2)
    wandb.define_metric('episode')
    wandb.define_metric('*', step_metric='episode')

    from mslog import TrainLog          # 2026-11-11 multi-seed: train_log.csv
    tlog = TrainLog(run_dir, env.env.unwrapped)
    csv_f = open(run_dir / 'episodes.csv', 'w', newline='')
    fields = ['episode', 'env_steps', 'Reward/Total', 'Reward/Total_MA100',
              'Success/Total_Delivered', 'Success/Total_Returned', 'Success/SR',
              'Success/SR_MA100', 'Loss/TD_loss', 'Grad_Norm', 'Epsilon',
              'Eval/Greedy_SR', 'Eval/Greedy_Reward']
    csv_w = csv.DictWriter(csv_f, fieldnames=fields, extrasaction='ignore')
    csv_w.writeheader()
    win_r, win_sr = deque(maxlen=100), deque(maxlen=100)
    best_r = best_sr = -float('inf')
    t_env = 0

    for ep_i in range(cfg['n_episodes']):
        obs = env.reset()
        tlog.start_episode()
        actor.reset()
        batch = partial(EpisodeBatch, scheme, groups, 1, cfg['episode_limit'] + 1,
                        preprocess=preprocess, device="cpu")()
        ep_r = 0.0
        dl = rt = 0
        step = 0
        done = False
        while not done and step < cfg['episode_limit']:
            avail = env.get_avail_actions()
            batch.update({"state": [env.get_state()], "avail_actions": [np.array(avail)],
                          "obs": [np.array(obs)]}, ts=step)
            actions = actor.act(obs, avail, t_env, test_mode=False)
            next_obs, rewards, dones, info = env.step(actions)
            team_r = float(sum(rewards))
            ep_r += team_r
            dl += info.get('step_delivery', 0)
            rt += info.get('step_return', 0)
            done = all(dones)
            # time limit only -> never a true terminal (same as the original mains)
            batch.update({"actions": [actions], "reward": [(team_r,)], "terminated": [(False,)]}, ts=step)
            obs = next_obs
            step += 1
            t_env += 1
        batch.update({"state": [env.get_state()], "avail_actions": [np.array(env.get_avail_actions())],
                      "obs": [np.array(obs)]}, ts=step)
        buffer.insert_episode_batch(batch)

        loss = gnorm = None
        if buffer.can_sample(cfg['batch_size']):
            sample = buffer.sample(cfg['batch_size'])
            if sample.device != cfg['device']:
                sample.to(cfg['device'])
            out = agent.update(sample, t_env)
            loss, gnorm = out[0], out[1]
            if ep_i % cfg['target_update_interval'] == 0:
                agent.update_targets()

        sr = dl + rt
        win_r.append(ep_r)
        win_sr.append(sr)
        log = {'episode': ep_i + 1, 'env_steps': t_env, 'Reward/Total': ep_r,
               'Reward/Total_MA100': float(np.mean(win_r)),
               'Success/Total_Delivered': dl, 'Success/Total_Returned': rt,
               'Success/SR': sr, 'Success/SR_MA100': float(np.mean(win_sr)),
               'Epsilon': float(agent.action_selector.epsilon)}
        if loss is not None:
            log['Loss/TD_loss'] = float(loss)
            log['Grad_Norm'] = float(gnorm)
        if a.greedy_eval_every and (ep_i + 1) % a.greedy_eval_every == 0:
            gr, gd, gt = play_greedy(eval_env, actor, cfg['episode_limit'])
            log['Eval/Greedy_SR'] = gd + gt
            log['Eval/Greedy_Reward'] = gr
        tlog.end_episode(ep_i + 1, t_env, dl, rt, log['Epsilon'])
        wandb.log(log)
        csv_w.writerow(log)
        csv_f.flush()

        if len(win_r) == win_r.maxlen:
            if log['Reward/Total_MA100'] > best_r:
                best_r = log['Reward/Total_MA100']
                agent.save(str(run_dir / 'best_model.pt'))
            if log['Success/SR_MA100'] > best_sr and a.save_best_sr:
                best_sr = log['Success/SR_MA100']
                agent.save(str(run_dir / 'best_sr_model.pt'))
        if (ep_i + 1) % 10 == 0:
            print('ep %6d/%d | R %8.2f | del %d ret %d SR %d (MA100 %.2f) | eps %.3f'
                  % (ep_i + 1, cfg['n_episodes'], ep_r, dl, rt, sr, log['Success/SR_MA100'],
                     log['Epsilon']), flush=True)

    agent.save(str(run_dir / 'model.pt'))
    csv_f.close()
    tlog.close()
    evals = {}
    for ck in ('best_sr_model.pt', 'best_model.pt', 'model.pt'):
        if (run_dir / ck).exists() and a.eval_episodes > 0:
            evals[ck] = evaluate_ckpt(run_dir / ck, eval_env, agent, actor, cfg, a.eval_episodes)
    wandb.summary.update({'eval/' + k.replace('.pt', '') + '/SR_mean': v['SR_mean'] for k, v in evals.items()})
    env.close()
    eval_env.close()
    wandb.finish()
    print('[done]', run_dir)


if __name__ == '__main__':
    main()

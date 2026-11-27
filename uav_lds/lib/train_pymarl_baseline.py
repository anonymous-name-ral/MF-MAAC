"""
QPLEX / QTRAN / QMIX training on UAV-LDS (same structure as MF-QMIX/train_mf_qmix.py).
Training loop of the original PyMARL-style scripts with:
  1. Unity engine at time scale 20, quality 1, no graphics, 100/100 parcels, 1000 steps.
  2. Delivery metrics from the environment (small / big deliveries, SR = small + 2 * big, collisions).
  3. --state_norm: running mean / std normalisation of the mixer input state (default off; on in the paper runs).
  4. CLI for epsilon_anneal_time, n_episodes, seed, worker id, wandb project / name, run dir.
Other hyper-parameters: rnn 128 + agent id, QPLEX mixing 64 / hypernet 128, QTRAN mixing 128, lr 5e-4,
buffer 1000 episodes, batch 32, target update 200, epsilon 1.0 -> 0.05.
  * --unity_seed (default = --seed) is passed to UnityEnvironment(seed=...)
  * episodes.csv holds each agent's episode reward (Reward/Agent_1..5; their sum = Reward/Episode_Raw) and Time/hours

  python train_pymarl_baseline.py --algo qplex --eps_anneal 10000000 --state_norm --worker 21 --run_dir results/<run>
"""
import argparse
import csv
import importlib.util
import json
import os
import random
import time
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb

ROOT = Path(__file__).resolve().parent
# Unity build: $UAVLDS_BUILD, default uav_lds/unity_build/Build_Linux_eval_v2_5uav/Logistics.x86_64
BUILD = Path(os.environ.get('UAVLDS_BUILD', ROOT.parent / 'unity_build' / 'Build_Linux_eval_v2_5uav' / 'Logistics.x86_64'))
Q_STAT_KEYS = ['Q/Q_tot_mean', 'Q/Q_tot_absmax', 'Q/Target_mean', 'Q/Qi_mean', 'Q/Qi_absmax', 'Q/TD_abs_mean']


def load_gym_env():
    """GymEnv with delivery counters (lib/UnityGymWrapperStats.py), loaded by file path."""
    spec = importlib.util.spec_from_file_location('UnityGymWrapperStats', ROOT / 'UnityGymWrapperStats.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GymEnv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def fmt_steps(n):
    return ('%gM' % (n / 1e6)) if n >= 1e6 else ('%gK' % (n / 1e3))


def run(config, a, Agent):
    set_seed(config["seed"])
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'config.json').exists():
        raise SystemExit('run_dir already holds a run: %s' % run_dir)

    GymEnv = load_gym_env()
    # engine setting "A" = original training setting of every UAV-LDS model
    env = GymEnv(name=str(BUILD), mapsize=13, numbuilding=3, max_smallbox=100, max_bigbox=100,
                 width=480, height=270, timescale=20, quality_level=1, no_graphics=True,
                 worker_id=a.worker, seed=config['unity_seed'])
    n_agents = env.nagent
    obs_dim = env.observation_space[0][0]
    n_actions = env.action_space_n[0]
    state_dim = obs_dim * n_agents
    config["unit_dim"] = obs_dim
    if a.algo == 'qplex':                              # as in main_qplex.py
        config.update(num_kernel=4, adv_hypernet_layers=1, adv_hypernet_embed=64)

    scheme = {
        "state": {"vshape": (state_dim,)},
        "obs": {"vshape": (obs_dim,), "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    if a.algo == 'qtran':                              # as in main_qtran.py
        scheme["actions_onehot"] = {"vshape": (n_actions,), "group": "agents", "dtype": torch.float}
    groups = {"agents": n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=n_actions)])}
    agent = Agent(config, obs_dim, n_actions, state_dim, n_agents)
    init_w = {k: v.detach().cpu().clone() for k, v in agent.agent.state_dict().items()}
    buffer = ReplayBuffer(scheme, groups, config["buffer_size"], config["episode_limit"] + 1,
                          config["burn_in_period"], preprocess=preprocess, device="cpu")

    name = '%s | eps %s | rs none%s' % (a.algo.upper(), fmt_steps(config["epsilon_anneal_time"]),
                                        ' | state_norm' if config["state_norm"] else '')
    if a.tag:
        name += ' | ' + a.tag
    config.update(run_dir=str(run_dir), build=str(BUILD), wandb_name=name, engine='A (training setting)')
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    wandb.init(entity=a.wandb_entity, project=a.wandb_project, name=name, group=a.algo.upper(), config=config,
               mode=a.wandb_mode, dir=str(run_dir))
    wandb.define_metric('episode')
    wandb.define_metric('*', step_metric='episode')

    t_env = 0
    max_avg_reward = -float('inf')
    hist = {'reward': [], 'SR': [], 'small': [], 'big': []}
    total = {'small': 0, 'big': 0}
    csv_f = open(run_dir / 'episodes.csv', 'w', newline='')
    writer = None
    model_tag = '%s_Logistics' % a.algo.upper()

    t0 = time.time()
    for ep_i in range(config["n_episodes"]):
        obs = env.reset()
        agent.init_hidden(batch_size=1)
        batch = partial(EpisodeBatch, scheme, groups, 1, config["episode_limit"] + 1,
                        preprocess=preprocess, device="cpu")()
        terminated = False
        episode_reward = 0.0
        agent_rewards = np.zeros(n_agents)
        step = 0
        info = {}
        while not terminated and step < config["episode_limit"]:
            avail_actions = env.get_avail_actions()
            batch.update({"state": [env.get_state()], "avail_actions": [np.array(avail_actions)],
                          "obs": [np.array(obs)]}, ts=step)
            actions = agent.step(obs, avail_actions, t_env, test_mode=False)
            next_obs, rewards, dones, info = env.step(np.eye(n_actions, dtype=np.int64)[actions])
            team_reward = float(sum(rewards))
            episode_reward += team_reward
            agent_rewards += np.asarray(rewards, dtype=np.float64)
            team_done = all(dones)
            terminated = team_done
            batch.update({"actions": [actions], "reward": [(team_reward,)],
                          "terminated": [(team_done != getattr(info, "episode_limit", False),)]}, ts=step)
            obs = next_obs
            step += 1
            t_env += 1
        batch.update({"state": [env.get_state()], "avail_actions": [np.array(env.get_avail_actions())],
                      "obs": [np.array(obs)]}, ts=step)
        buffer.insert_episode_batch(batch)

        loss = grad_norm = None
        if buffer.can_sample(config["batch_size"]):
            sample = buffer.sample(config["batch_size"])
            if sample.device != config["device"]:
                sample.to(config["device"])
            loss, grad_norm = agent.update(sample, t_env)
            if ep_i % config["target_update_interval"] == 0:
                agent.update_targets()

        small, big = info.get('small_delivered', 0), info.get('big_delivered', 0)
        total['small'] += small
        total['big'] += big
        for k, v in (('reward', episode_reward), ('SR', small + 2 * big), ('small', small), ('big', big)):
            hist[k].append(v)
        ma = {k: float(np.mean(v[-100:])) for k, v in hist.items()}
        if len(hist['reward']) >= 100 and ma['reward'] > max_avg_reward:
            max_avg_reward = ma['reward']
            agent.save(str(run_dir / 'best_model.pt'))
        log = {'episode': ep_i + 1, 'Global_Step': t_env, 'Epsilon': agent.action_selector.epsilon,
               'Reward/Episode_Raw': episode_reward, 'Reward/Episode_Avg_100': ma['reward'],
               'Success/Small': small, 'Success/Big': big, 'Success/SR': small + 2 * big,
               'Success/SR_MA100': ma['SR'], 'Success/Small_MA100': ma['small'], 'Success/Big_MA100': ma['big'],
               'Success/Small_Total': total['small'], 'Success/Big_Total': total['big'],
               'Collisions': info.get('collisions', 0), 'Time/hours': (time.time() - t0) / 3600.0}
        log.update({'Reward/Agent_%d' % (i + 1): float(r) for i, r in enumerate(agent_rewards)})
        if loss is not None:
            log['Loss/TD'] = float(loss)
            log['Grad_Norm'] = float(grad_norm)
            log.update(agent.last_q_stats)
        wandb.log(log)
        if writer is None:
            writer = csv.DictWriter(csv_f, fieldnames=list(log.keys()) + ['Loss/TD', 'Grad_Norm'] + Q_STAT_KEYS,
                                    extrasaction='ignore')
            writer.writeheader()
        writer.writerow(log)
        if (ep_i + 1) % 10 == 0:
            csv_f.flush()
            print('Ep %6d/%d | R %8.2f (MA %8.2f) | small %2d big %2d SR %3d (MA %.2f) | eps %.3f | loss %s'
                  % (ep_i + 1, config["n_episodes"], episode_reward, ma['reward'], small, big, small + 2 * big,
                     ma['SR'], agent.action_selector.epsilon, '-' if loss is None else '%.4f' % float(loss)),
                  flush=True)

    csv_f.close()
    env.close()
    agent.save(str(run_dir / 'final_model.pt'))
    changed = any(not torch.equal(init_w[k], v.detach().cpu()) for k, v in agent.agent.state_dict().items())
    print('[check] agent weights changed during training:', changed, flush=True)
    with open(run_dir / ('%s_rewards.json' % model_tag), 'w') as f:
        json.dump(hist['reward'], f)
    wandb.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['qplex', 'qtran', 'qmix'], required=True)
    p.add_argument('--eps_anneal', type=int, required=True)
    p.add_argument('--state_norm', action='store_true', help='running mean/std normalisation of the mixer input')
    p.add_argument('--n_episodes', type=int, default=60000)
    p.add_argument('--seed', type=int, default=40)
    p.add_argument('--unity_seed', type=int, default=None, help='Unity RNG seed; default = --seed')
    p.add_argument('--worker', type=int, required=True, help='unique per concurrent run (port offset)')
    p.add_argument('--run_dir', required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--burn_in', type=int, default=32, help='episodes before learning starts (smoke tests only)')
    p.add_argument('--wandb_project', default='MF-MAAC_UAV-LDS')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--wandb_mode', default='disabled')
    p.add_argument('--tag', default='', help='appended to the wandb name')
    a = p.parse_args()

    FAMILY_DIR = {'qplex': 'QPLEX', 'qtran': 'QTRAN', 'qmix': 'QMIX'}[a.algo]
    sys.path.insert(0, str(ROOT / FAMILY_DIR))             # QPLEX/, QTRAN/ or QMIX/ packages (modules, components, ...)
    from components.episode_buffer import EpisodeBatch, ReplayBuffer  # noqa: E402
    from components.transforms import OneHot  # noqa: E402
    if a.algo == 'qplex':
        from algorithms.qplex_agent import QPLEXAgent as Agent  # noqa: E402
    elif a.algo == 'qtran':
        from algorithms.qtran_agent import QTRANAgent as Agent  # noqa: E402
    else:
        from algorithms.qmix_agent import QMIXAgent as Agent  # noqa: E402

    # identical to main_qplex.py / main_qtran.py __main__ except the CLI-controlled fields
    config = {
        "env_id": "Logistics_Multi_Env", "model_name": "%s_Logistics" % a.algo.upper(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_episodes": a.n_episodes, "episode_limit": 1000, "seed": a.seed,
        "unity_seed": a.seed if a.unity_seed is None else a.unity_seed,
        "buffer_size": 1000, "batch_size": a.batch_size, "burn_in_period": a.burn_in,
        "lr": 0.0005, "optim_alpha": 0.99, "optim_eps": 0.00001, "gamma": 0.99, "grad_norm_clip": 10,
        "target_update_interval": 200, "double_q": True,
        "action_selector": "epsilon_greedy", "epsilon_start": 1.0, "epsilon_finish": 0.05,
        "epsilon_anneal_time": a.eps_anneal,
        "rnn_hidden_dim": 128,
        "state_norm": a.state_norm,
    }
    if a.algo == 'qplex':
        config.update({"mixing_embed_dim": 64, "hypernet_embed": 128, "n_head": 4, "attend_reg_coef": 0.001,
                       "nonlinear": False, "weighted_head": True, "mask_dead": False, "state_bias": True,
                       "is_minus_one": True})
    elif a.algo == 'qtran':
        config.update({"mixing_embed_dim": 128, "qtran_arch": "qtran_paper", "network_size": "small",
                       "opt_loss": 1.0, "nopt_min_loss": 0.1})
    else:   # standard QMIX mixer, 1-layer hypernets as MF-QMIX; mixing 64 as QPLEX/MF-QMIX
        config.update({"mixer": "qmix", "mixing_embed_dim": 64, "hypernet_layers": 1})
    run(config, a, Agent)

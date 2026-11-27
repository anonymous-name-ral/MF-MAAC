"""
MAPPO runner for RWARE.

The rollout / train loop and the config are copied from the original MAPPO main script
(algorithm code in MAPPO/ is untouched). Added: CLI, reward_mode switch with
beta, per-episode CSV metrics, best-by-SR
checkpoint, and an optional final evaluation (stochastic and deterministic).

Example:
  python train_mappo.py --reward_mode custom --reward_beta 0.1 --seed 40 --n_episodes 40000 --run_dir results/<run>
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'MAPPO'))
ENV_IDS = {'custom': 'rware-small-5ag-twostage-v2',
           'event': 'rware-small-5ag-twostage-event-v2',
           'original': 'rware-small-5ag-twostage-orig-v2'}

# copied from MAPPO/main_mappo.py (__main__)
CONFIG = {
    "model_name": "MAPPO_RWARE", "seed": 40, "n_rollout_threads": 1, "n_episodes": 40000,
    "episode_length": 500, "use_gpu": True, "render_mode": 'rgb_array', "render_fps": 30,
    "algorithm_name": "rmappo", "hidden_size": 128, "layer_N": 2, "use_ReLU": True,
    "use_feature_normalization": True, "use_orthogonal": True,
    "lr": 5e-4, "critic_lr": 5e-4, "opti_eps": 1e-5, "weight_decay": 0, "max_grad_norm": 10.0,
    "use_max_grad_norm": True, "gain": 0.01,
    "ppo_epoch": 10, "num_mini_batch": 1, "clip_param": 0.2, "value_loss_coef": 1.0,
    "entropy_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95, "use_gae": True,
    "use_valuenorm": True, "use_popart": False, "use_huber_loss": True, "huber_delta": 10.0,
    "use_clipped_value_loss": True,
    "use_proper_time_limits": False, "use_value_active_masks": False,
    "use_policy_active_masks": False, "stacked_frames": 1,
}


def set_up_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class RWARE_MAPPO_Wrapper:
    """Same as MAPPO/main_mappo.py, plus reward_beta override."""
    def __init__(self, env_id, seed=None, render_mode='rgb_array', reward_beta=None):
        import gym
        import rware  # noqa: F401
        self.env = gym.make(env_id, render_mode=render_mode)
        if seed is not None:
            self.env.seed(seed)
        if reward_beta is not None:
            self.env.unwrapped.reward_beta = reward_beta
        self.nagent = self.env.n_agents
        self.observation_space = [(space.shape[0],) for space in self.env.observation_space]
        self.action_space = [space.n for space in self.env.action_space]

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return [np.array(o, dtype=np.float32) for o in obs]

    def step(self, actions):
        if isinstance(actions[0], (np.ndarray, list, torch.Tensor)):
            discrete_actions = [int(act[0]) if hasattr(act, '__len__') else int(act) for act in actions]
        else:
            discrete_actions = [int(act) for act in actions]
        step_result = self.env.step(discrete_actions)
        if len(step_result) == 5:
            next_obs, rewards, dones, truncated, infos = step_result
        else:
            next_obs, rewards, dones, infos = step_result
        return [np.array(o, dtype=np.float32) for o in next_obs], rewards, dones, infos

    def close(self):
        self.env.close()


def evaluate(policy, cfg, ckpt, n_episodes, deterministic, base_seed=10000):
    sd = torch.load(ckpt, map_location=policy.device)
    policy.actor.load_state_dict(sd['actor'])
    policy.actor.eval()
    env = RWARE_MAPPO_Wrapper(cfg['env_id'], reward_beta=cfg['reward_beta'])
    n = env.nagent
    rows = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        set_up_seed(seed)
        env.env.seed(seed)
        obs = np.array(env.reset(), dtype=np.float32)
        rew = dl = rt = 0.0
        for _ in range(cfg['episode_length']):
            with torch.no_grad():
                act = policy.act(obs, deterministic=deterministic)
            obs, r, dones, info = env.step(act.cpu().numpy())
            obs = np.array(obs, dtype=np.float32)
            rew += float(sum(r))
            dl += info.get('step_delivery', 0)
            rt += info.get('step_return', 0)
            if all(dones):
                break
        rows.append({'seed': seed, 'delivered': dl, 'returned': rt, 'SR': dl + rt, 'reward': rew})
    env.close()
    a = lambda k: np.array([x[k] for x in rows], dtype=float)
    summary = {'model': Path(ckpt).name, 'algo': cfg['model'], 'reward_mode': cfg['reward_mode'],
               'seed': cfg['seed'], 'n_episodes': n_episodes, 'base_seed': base_seed,
               'explore': not deterministic, 'SR_mean': a('SR').mean(), 'SR_std': a('SR').std(),
               'ATCD': a('delivered').mean(), 'ASR': a('returned').mean(), 'reward_mean': a('reward').mean()}
    suffix = '.eval.json' if deterministic else '.eval_stochastic.json'
    with open(Path(ckpt).with_suffix('').as_posix() + suffix, 'w') as f:
        json.dump({'summary': summary, 'episodes': rows}, f, indent=2)
    print('[eval] %s %s: SR %.2f +- %.2f (ATCD %.2f, ASR %.2f)'
          % (Path(ckpt).name, 'deterministic' if deterministic else 'stochastic',
             summary['SR_mean'], summary['SR_std'], summary['ATCD'], summary['ASR']))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reward_mode', default='event', choices=list(ENV_IDS))
    p.add_argument('--reward_beta', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=40)
    p.add_argument('--n_episodes', type=int, default=40000)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--save_best_sr', type=int, default=1)
    p.add_argument('--wandb_mode', default='disabled')
    p.add_argument('--wandb_project', default='MF-MAAC_RWARE')
    p.add_argument('--wandb_entity', default=None)
    a = p.parse_args()

    from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
    from utils.shared_buffer import SharedReplayBuffer
    from algorithms.rMAPPOPolicy import R_MAPPOPolicy
    from algorithms.r_mappo import R_MAPPO

    cfg = dict(CONFIG)
    cfg.update({'seed': a.seed, 'n_episodes': a.n_episodes, 'reward_mode': a.reward_mode,
                'env_id': ENV_IDS[a.reward_mode],
                'reward_beta': a.reward_beta if a.reward_mode != 'original' else None,
                'model': 'MAPPO'})
    set_up_seed(cfg['seed'])
    args = SimpleNamespace(**cfg)
    device = torch.device("cuda:0" if cfg["use_gpu"] and torch.cuda.is_available() else "cpu")

    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'config.json').exists():
        raise SystemExit('run_dir already holds a run: %s' % run_dir)

    E = cfg['n_rollout_threads']
    def get_env_fn(rank):
        def init_env():
            return RWARE_MAPPO_Wrapper(cfg['env_id'], seed=cfg['seed'], render_mode=cfg['render_mode'],
                                       reward_beta=cfg['reward_beta'])
        return init_env
    envs = DummyVecEnv([get_env_fn(0)]) if E == 1 else SubprocVecEnv([get_env_fn(i) for i in range(E)])
    num_agents = envs.nagent
    obs_dim = envs.observation_space[0][0]
    act_dim = envs.action_space[0]

    class Box:
        def __init__(self, shape):
            self.shape = shape

    class Discrete:
        def __init__(self, n):
            self.n = n

    policy = R_MAPPOPolicy(args, Box((obs_dim,)), Box((obs_dim * num_agents,)), Discrete(act_dim), device=device)
    trainer = R_MAPPO(args, policy, device=device)
    buffer = SharedReplayBuffer(args, num_agents, Box((obs_dim,)), Box((obs_dim * num_agents,)), Discrete(act_dim))

    sys.path.insert(0, str(HERE))
    from wandb_naming import run_name
    import wandb
    wb_name, cfg['model_version'] = run_name(cfg['model'], cfg['reward_mode'], consume=a.wandb_mode == 'online')
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    wandb.init(entity=a.wandb_entity, project=a.wandb_project, name=wb_name, group=cfg['model'],
               tags=[cfg['model'], cfg['reward_mode']], config=cfg, mode=a.wandb_mode, dir=str(run_dir))
    wandb.define_metric('episode')
    wandb.define_metric('*', step_metric='episode')

    from mslog import TrainLog          # train_log.csv
    tlog = TrainLog(run_dir, envs.envs[0].env.unwrapped)
    csv_f = open(run_dir / 'episodes.csv', 'w', newline='')
    fields = ['episode', 'env_steps', 'Reward/Total', 'Reward/Total_MA100', 'Success/Total_Delivered',
              'Success/Total_Returned', 'Success/SR', 'Success/SR_MA100', 'Loss/Critic_loss',
              'Loss/Mean_Actor_Loss', 'Metrics/Dist_Entropy']
    csv_w = csv.DictWriter(csv_f, fieldnames=fields, extrasaction='ignore')
    csv_w.writeheader()
    win_r, win_sr = deque(maxlen=100), deque(maxlen=100)
    best_r = best_sr = -float('inf')
    t = 0
    save = lambda path: torch.save({'actor': policy.actor.state_dict(), 'critic': policy.critic.state_dict()}, path)

    for ep_i in range(0, cfg["n_episodes"], E):
        obs = np.array(envs.reset(), dtype=np.float32)
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=0)
        concat_obs = obs.reshape(E, -1)
        share_obs = np.repeat(concat_obs[:, np.newaxis, :], num_agents, axis=1)
        buffer.share_obs[0] = share_obs.copy()
        buffer.obs[0] = obs.copy()
        trainer.prep_rollout()
        tlog.start_episode()
        total_reward = 0.0
        total_delivery = total_return = 0

        for step in range(cfg["episode_length"]):
            with torch.no_grad():
                value, action, action_log_prob = policy.get_actions(
                    np.concatenate(buffer.share_obs[step]), np.concatenate(buffer.obs[step]))
            values = np.array(np.array_split(value.cpu().numpy(), E))
            actions = np.array(np.array_split(action.cpu().numpy(), E))
            action_log_probs = np.array(np.array_split(action_log_prob.cpu().numpy(), E))

            env_actions = actions.squeeze(-1) if actions.shape[-1] == 1 else actions
            next_obs, rewards, dones, infos = envs.step(env_actions)
            t += E
            next_obs = np.array(next_obs, dtype=np.float32)
            if next_obs.ndim == 2:
                next_obs = np.expand_dims(next_obs, axis=0)
            rewards = np.array(rewards, dtype=np.float32)
            if rewards.ndim == 2:
                rewards = np.expand_dims(rewards, axis=-1)
            masks = np.ones((E, num_agents, 1), dtype=np.float32)
            if isinstance(dones, (list, np.ndarray)):
                for th_i, d in enumerate(dones):
                    if np.all(d):
                        masks[th_i] = np.zeros((num_agents, 1), dtype=np.float32)
            next_concat_obs = next_obs.reshape(E, -1)
            next_share_obs = np.repeat(next_concat_obs[:, np.newaxis, :], num_agents, axis=1)
            buffer.insert(next_share_obs, next_obs, actions, action_log_probs, values, rewards, masks)
            total_reward += float(rewards[0].sum())
            total_delivery += infos[0].get("step_delivery", 0)
            total_return += infos[0].get("step_return", 0)

        trainer.prep_training()
        with torch.no_grad():
            next_value = policy.get_values(np.concatenate(buffer.share_obs[-1]))
            next_value = np.array(np.array_split(next_value.cpu().numpy(), E))
        buffer.compute_returns(next_value, trainer.value_normalizer)
        train_info = trainer.train(buffer)
        buffer.after_update()

        sr = total_delivery + total_return
        win_r.append(total_reward)
        win_sr.append(sr)
        log = {'episode': ep_i + E, 'env_steps': t, 'Reward/Total': total_reward,
               'Reward/Total_MA100': float(np.mean(win_r)),
               'Success/Total_Delivered': total_delivery, 'Success/Total_Returned': total_return,
               'Success/SR': sr, 'Success/SR_MA100': float(np.mean(win_sr)),
               'Loss/Critic_loss': float(train_info.get('value_loss', 0)),
               'Loss/Mean_Actor_Loss': float(train_info.get('policy_loss', 0)),
               'Metrics/Dist_Entropy': float(train_info.get('dist_entropy', 0))}
        tlog.end_episode(ep_i + E, t, total_delivery, total_return)
        wandb.log(log)
        csv_w.writerow(log)
        csv_f.flush()
        if len(win_r) == win_r.maxlen:
            if log['Reward/Total_MA100'] > best_r:
                best_r = log['Reward/Total_MA100']
                save(run_dir / 'best_model.pt')
            if log['Success/SR_MA100'] > best_sr and a.save_best_sr:
                best_sr = log['Success/SR_MA100']
                save(run_dir / 'best_sr_model.pt')
        if (ep_i + E) % 10 == 0:
            print('ep %6d/%d | R %8.2f | del %d ret %d SR %d (MA100 %.2f) | vloss %.4f'
                  % (ep_i + E, cfg['n_episodes'], total_reward, total_delivery, total_return, sr,
                     log['Success/SR_MA100'], log['Loss/Critic_loss']), flush=True)

    save(run_dir / 'model.pt')
    csv_f.close()
    tlog.close()
    envs.close()
    evals = {}
    for ck in ('best_sr_model.pt', 'best_model.pt', 'model.pt'):
        if (run_dir / ck).exists() and a.eval_episodes > 0:
            evals[ck] = evaluate(policy, cfg, run_dir / ck, a.eval_episodes, deterministic=False)
            evaluate(policy, cfg, run_dir / ck, a.eval_episodes, deterministic=True)
    wandb.summary.update({'eval/' + k.replace('.pt', '') + '/SR_mean_stochastic': v['SR_mean']
                          for k, v in evals.items()})
    wandb.finish()
    print('[done]', run_dir)


if __name__ == '__main__':
    main()

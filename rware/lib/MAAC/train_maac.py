"""
RWARE training of the 15 MAAC-family models (MAAC, F-MAAC, MAHAC backbones and their MF / MSF variants).
  * --critic_type selects the critic module (CRITICS below)
  * environment seed = --seed; reward weight of the team term set by --reward_beta; results in --run_dir
  * train_log.csv per episode (../mslog.py)
Hyper-parameters (values of `config` in __main__): 1 rollout thread, buffer 1e6 steps, 500-step episodes,
2 updates every 200 steps, batch 1024, hidden 128, 4 attention heads, lr 1e-3 / 1e-3, tau 1e-3, gamma 0.99,
reward scale 7. best_model.pt = best 100-episode moving average of the episode team reward, model.pt = final.

  python train_maac.py --critic_type mf_p --seed 40 --run_dir <dir>
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable

HERE = Path(__file__).resolve().parent
CRITICS = {
    'plain': ('utils.critics_maac', 'MAAC'),
    'mf_p': ('utils.critics_mf_maac_p', 'MF-MAAC-P'),
    'mf_t': ('utils.critics_mf_maac_t', 'MF-MAAC-T'),
    'msf_p': ('utils.critics_msf_maac_p', 'MSF-MAAC-P'),
    'msf_t': ('utils.critics_msf_maac_t', 'MSF-MAAC-T'),
    'mahac': ('utils.critics_mahac', 'MAHAC'),
    'mahac_mf_p': ('utils.critics_mf_mahac_p', 'MF-MAHAC-P'),
    'mahac_mf_t': ('utils.critics_mf_mahac_t', 'MF-MAHAC-T'),
    'mahac_msf_p': ('utils.critics_msf_mahac_p', 'MSF-MAHAC-P'),
    'mahac_msf_t': ('utils.critics_msf_mahac_t', 'MSF-MAHAC-T'),
    'fmaac': ('utils.critics_f_maac', 'F-MAAC'),
    'fmaac_mf_p': ('utils.critics_mf_f_maac_p', 'MF-F-MAAC-P'),
    'fmaac_mf_t': ('utils.critics_mf_f_maac_t', 'MF-F-MAAC-T'),
    'fmaac_msf_p': ('utils.critics_msf_f_maac_p', 'MSF-F-MAAC-P'),
    'fmaac_msf_t': ('utils.critics_msf_f_maac_t', 'MSF-F-MAAC-T'),
}


def set_up_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class RWARE_Wrapper:
    def __init__(self, env_id, seed=None, render_mode='human', render_fps=10):
        self.env = gym.make(env_id, render_mode=render_mode)
        if seed is not None:
            self.env.seed(seed)
        if hasattr(self.env.unwrapped, 'metadata'):
            self.env.unwrapped.metadata['render_fps'] = render_fps
        self.nagent = self.env.n_agents
        self.observation_space = [(space.shape[0],) for space in self.env.observation_space]
        self.action_space = [space.n for space in self.env.action_space]

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return [np.array(o) for o in obs]

    def step(self, actions):
        discreate_actions = [int(np.argmax(act)) for act in actions]
        step_result = self.env.step(discreate_actions)
        if len(step_result) == 5:
            next_obs, rewards, dones, truncated, infos = step_result
        else:
            next_obs, rewards, dones, infos = step_result
        return [np.array(o) for o in next_obs], rewards, dones, infos

    def close(self):
        self.env.close()


def make_parallel_env(env_id, n_rollout_threads, seed, render_mode, render_fps):
    def get_env_fn(rank):
        def init_env():
            return RWARE_Wrapper(env_id=env_id, seed=seed, render_mode=render_mode, render_fps=render_fps)
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    set_up_seed(config["seed"])
    run_dir = Path(config["run_dir"])
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(str(log_dir))
    wandb.init(mode="disabled")

    env = make_parallel_env(config["env_id"], config["n_rollout_threads"], config["seed"],
                            config["render_mode"], config["render_fps"])
    wh = env.envs[0].env.unwrapped
    assert wh.reward_mode == 'custom'
    wh.reward_beta = config["reward_beta"]
    model = AttentionSAC.init_from_env(env, tau=config["tau"], pi_lr=config["pi_lr"], q_lr=config["q_lr"],
                                       gamma=config["gamma"], pol_hidden_dim=config["pol_hidden_dim"],
                                       critic_hidden_dim=config["critic_hidden_dim"],
                                       attend_heads=config["attend_heads"], reward_scale=config["reward_scale"])
    replay_buffer = ReplayBuffer(config["buffer_length"], model.nagents,
                                 [obsp[0] for obsp in env.observation_space],
                                 [acsp for acsp in env.action_space])
    tlog = TrainLog(run_dir, wh)

    t = 0
    episode_total_rewards = []
    max_avg_reward = -float('inf')
    window_size = 100

    for ep_i in range(0, config["n_episodes"], config["n_rollout_threads"]):
        obs = env.reset()
        tlog.start_episode()
        model.prep_rollouts(device='cpu')
        episode_rewards = [0] * env.nagent
        total_reward = 0
        total_delivery = 0
        total_return = 0

        for step in range(config["episode_length"]):
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False)
                         for i in range(model.nagents)]
            torch_agent_actions = model.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = np.array([[ac[i] for ac in agent_actions] for i in range(config["n_rollout_threads"])])

            next_obs, rewards, done, info = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, done)
            obs = next_obs

            for i in range(env.nagent):
                episode_rewards[i] += rewards[0][i]
            total_reward += sum(rewards[0])
            total_delivery += info[0].get("step_delivery", 0)
            total_return += info[0].get("step_return", 0)
            t += config["n_rollout_threads"]

            if (len(replay_buffer) >= config["batch_size"] and
                    (t % config["steps_per_update"]) < config["n_rollout_threads"]):
                model.prep_training(device='gpu' if config["use_gpu"] else 'cpu')
                for u_i in range(config["num_updates"]):
                    sample = replay_buffer.sample(config["batch_size"], to_gpu=config["use_gpu"])
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')

        episode_total_rewards.append(total_reward)
        tlog.end_episode(ep_i + 1, t, total_delivery, total_return)
        if (ep_i + 1) % 10 == 0:
            print("episode : %8d/%8d | R %8.2f | del %d ret %d" %
                  (ep_i + 1, config["n_episodes"], total_reward, total_delivery, total_return), flush=True)

        ep_rews = replay_buffer.get_average_rewards(config["episode_length"] * config["n_rollout_threads"])
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew * config["episode_length"], ep_i)

        if len(episode_total_rewards) >= window_size:
            current_avg_reward = np.mean(episode_total_rewards[-window_size:])
            if current_avg_reward > max_avg_reward:
                max_avg_reward = current_avg_reward
                model.prep_rollouts(device='cpu')
                model.save(run_dir / 'best_model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    tlog.close()
    logger.close()
    with open(run_dir / 'total_episode_rewards.json', 'w') as f:
        json.dump([float(r) for r in episode_total_rewards], f)
    print('[done]', run_dir, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--critic_type', required=True, choices=list(CRITICS))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--reward_beta', type=float, default=0.1)
    p.add_argument('--n_episodes', type=int, default=40000)
    p.add_argument('--run_dir', required=True)
    a = p.parse_args()
    os.environ["MS_CRITIC_MODULE"] = CRITICS[a.critic_type][0]

    sys.path.insert(0, str(HERE.parent))
    import gym
    import rware  # noqa: F401  (registers the environments)
    import wandb
    from tensorboardX import SummaryWriter
    from utils.buffer import ReplayBuffer
    from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
    from algorithms.attention_sac_db import AttentionSAC
    from mslog import TrainLog

    # training configuration of the paper runs
    config = dict(env_id="rware-small-5ag-twostage-v2", n_rollout_threads=1, buffer_length=int(1e6),
                  episode_length=500, steps_per_update=200, num_updates=2, batch_size=1024,
                  pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, pi_lr=0.001, q_lr=0.001,
                  tau=0.001, gamma=0.99, reward_scale=7.0, use_gpu=True, render_mode='rgb_array', render_fps=30)
    config.update(model=CRITICS[a.critic_type][1], critic_type=a.critic_type,
                  critic_module=CRITICS[a.critic_type][0], seed=a.seed, env_seed=a.seed,
                  reward_mode='custom', reward_beta=a.reward_beta, n_episodes=a.n_episodes,
                  run_dir=a.run_dir, code='MAAC/train_maac.py')
    Path(a.run_dir).mkdir(parents=True, exist_ok=True)
    if (Path(a.run_dir) / 'config.json').exists():
        raise SystemExit('run_dir already holds a run: %s' % a.run_dir)
    with open(Path(a.run_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    run(config)

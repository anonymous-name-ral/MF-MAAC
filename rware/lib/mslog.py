"""
Per-episode training log: <run_dir>/train_log.csv.

Pure bookkeeping (no random numbers are drawn), so the training run is unchanged by it.
Warehouse._compute_reward is wrapped to sum, per agent and episode, the reward the learner
receives and the team term beta * sum(base) inside it:
  team_reward_train = sum over agents of the reward used for training (incl. the team term)
  team_reward       = the same without the team term (same definition for every algorithm)
  agent_i           = agent i's reward without the team term (sums to team_reward)
"""
import csv
import time

import numpy as np
from rware.warehouse import Warehouse

_compute_reward_orig = Warehouse._compute_reward


def _compute_reward(self, base_rewards, task_rewards, failed_agents):
    out = _compute_reward_orig(self, base_rewards, task_rewards, failed_agents)
    if getattr(self, '_ms_rew', None) is None:
        self._ms_rew, self._ms_team = np.zeros(self.n_agents), np.zeros(self.n_agents)
    self._ms_rew += np.asarray(out, dtype=float)
    if self.reward_mode != 'original':
        self._ms_team += self.reward_beta * float(np.sum(base_rewards))
    return out


Warehouse._compute_reward = _compute_reward


class TrainLog(object):
    def __init__(self, run_dir, env):
        """env: the training Warehouse instance (env.unwrapped)"""
        self.env, self.n, self.t0 = env, env.n_agents, time.time()
        self.fields = (['episode', 'env_steps', 'elapsed_sec', 'team_reward', 'team_reward_train']
                       + ['agent_%d' % i for i in range(self.n)]
                       + ['delivered', 'returned', 'AS', 'epsilon'])
        self.f = open(str(run_dir) + '/train_log.csv', 'w', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=self.fields)
        self.w.writeheader()

    def start_episode(self):
        self.env._ms_rew, self.env._ms_team = np.zeros(self.n), np.zeros(self.n)

    def end_episode(self, episode, env_steps, delivered, returned, epsilon=None):
        own = self.env._ms_rew - self.env._ms_team
        row = {'episode': episode, 'env_steps': env_steps, 'elapsed_sec': round(time.time() - self.t0, 1),
               'team_reward': float(own.sum()), 'team_reward_train': float(self.env._ms_rew.sum()),
               'delivered': int(delivered), 'returned': int(returned), 'AS': int(delivered + returned),
               'epsilon': '' if epsilon is None else float(epsilon)}
        row.update({'agent_%d' % i: float(own[i]) for i in range(self.n)})
        self.w.writerow(row)
        self.f.flush()
        return row

    def close(self):
        self.f.close()

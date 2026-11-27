"""
2026-10-22 QMIX with the authors' official setting (PyMARL, Rashid et al. 2018; default.yaml / qmix.yaml):
  agent input = [own observation, own previous action one-hot (zeros at t=0), agent id one-hot]
                (obs_last_action=True, obs_agent_id=True), one shared RNN agent (fc -> GRUCell -> fc)
  mixer       = QMixer on the global state with 2-layer hypernetworks (hypernet_layers=2)
  learner     = PyMARL QLearner: double Q, TD(0) on the masked episode batch, RMSprop, grad clip, hard target update
Training sizes / schedule come from train_pymarl_baseline.py (same as the MF-QMIX run they are compared with).
Same interface as MFQMIXAgent: step(obs, last_action_onehot, avail, t_env), update(batch, t_env), save().
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.optim import RMSprop

from modules.mixers.qmix_ import QMixer
from components.action_selectors import REGISTRY as action_REGISTRY


class RNNAgentPyMARL(nn.Module):
    """PyMARL modules/agents/rnn_agent.py; input_shape already includes last action and agent id."""
    def __init__(self, input_shape, args):
        super(RNNAgentPyMARL, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        orig = inputs.shape[:-1]
        x = F.relu(self.fc1(inputs.reshape(-1, inputs.shape[-1])))
        h = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))
        q = self.fc2(h)
        return q.view(*orig, -1), h.view(*orig, -1)


class QMIXPyMARLAgent(object):
    def __init__(self, config, obs_shape, n_actions, state_dim, n_agents):
        self.args = SimpleNamespace(**config)
        self.args.n_agents = n_agents
        self.args.n_actions = n_actions
        self.args.state_shape = [state_dim]
        self.args.hypernet_layers = 2
        self.n_agents, self.n_actions = n_agents, n_actions
        self.agent = RNNAgentPyMARL(obs_shape + n_actions + n_agents, self.args).to(self.args.device)
        self.mixer = QMixer(self.args).to(self.args.device)
        self.target_agent = copy.deepcopy(self.agent)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = RMSprop(params=self.params, lr=self.args.lr,
                                 alpha=self.args.optim_alpha, eps=self.args.optim_eps)
        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)
        self.hidden_states = None

    def init_hidden(self, batch_size=1):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def _inputs(self, obs, last_action_onehot):
        """obs (bs, n, od), last action (bs, n, na) -> (bs, n, od + na + n)"""
        bs = obs.shape[0]
        ids = torch.eye(self.n_agents, device=obs.device).unsqueeze(0).expand(bs, -1, -1)
        return torch.cat([obs, last_action_onehot, ids], dim=-1)

    def step(self, obs, last_action_onehot, avail_actions, t_env, test_mode=False):
        o = torch.tensor(obs, dtype=torch.float32, device=self.args.device).unsqueeze(0)
        la = torch.tensor(last_action_onehot, dtype=torch.float32, device=self.args.device).unsqueeze(0)
        agent_outs, self.hidden_states = self.agent(self._inputs(o, la), self.hidden_states)
        avail = torch.tensor(avail_actions, dtype=torch.float32, device=self.args.device).unsqueeze(0)
        return self.action_selector.select_action(agent_outs, avail, t_env, test_mode=test_mode).squeeze(0).cpu().numpy()

    def update(self, batch, t_env):
        if batch.device != self.args.device:
            batch.to(self.args.device)
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        obs, state = batch["obs"], batch["state"]
        bs, T = obs.shape[0], obs.shape[1]
        last_actions = torch.cat([torch.zeros_like(actions_onehot[:, 0:1]), actions_onehot[:, :-1]], dim=1)

        mac_out = []
        self.init_hidden(bs)
        for t in range(T):
            q, self.hidden_states = self.agent(self._inputs(obs[:, t], last_actions[:, t]), self.hidden_states)
            mac_out.append(q)
        mac_out = torch.stack(mac_out, dim=1)                                   # (bs, T, n, na)
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        th = self.target_agent.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)
        for t in range(T):
            q, th = self.target_agent(self._inputs(obs[:, t], last_actions[:, t]), th)
            target_mac_out.append(q)
        target_mac_out = torch.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        if getattr(self.args, "double_q", True):
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        qi_raw = chosen_action_qvals.clone().detach()
        chosen_action_qvals = self.mixer(chosen_action_qvals, state[:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, state[:, 1:])
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, getattr(self.args, "grad_norm_clip", 10))
        self.optimizer.step()
        with torch.no_grad():
            stats = {"loss": loss.item(), "grad_norm": grad_norm.item(),
                     "td_error_abs_mean": masked_td_error.abs().sum().item() / mask.sum().item(),
                     "q_tot_mean": chosen_action_qvals.mean().item(), "target_q_mean": targets.mean().item(),
                     "qi_mean": qi_raw.mean().item(), "qi_std": qi_raw.std().item()}
        return loss.item(), grad_norm.item(), stats

    def update_targets(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, filename):
        torch.save({'agent': self.agent.state_dict(), 'mixer': self.mixer.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, filename)

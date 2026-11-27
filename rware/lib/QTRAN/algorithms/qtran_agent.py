import torch
import copy
from types import SimpleNamespace
from torch.optim import RMSprop

# Import QTRAN core components
from modules.agents.rnn_agent import RNNAgent
from modules.mixers.qtran import QTranBase
from components.action_selectors import REGISTRY as action_REGISTRY


class QTRANAgent(object):
    def __init__(self, config, obs_shape, n_actions, state_dim, n_agents):
        self.args = SimpleNamespace(**config)
        self.args.n_agents = n_agents
        self.args.n_actions = n_actions
        self.args.state_shape = [state_dim]

        self.n_agents = n_agents
        self.n_actions = n_actions

        # 1. Agent initialization[cite: 1]
        self.agent = RNNAgent(obs_shape, self.args).to(self.args.device)

        # 2. Mixer initialization (QTRAN)[cite: 4]
        self.mixer = QTranBase(self.args).to(self.args.device)

        # 3. Copy target networks[cite: 1]
        self.target_agent = copy.deepcopy(self.agent)
        self.target_mixer = copy.deepcopy(self.mixer)

        # 4. Optimizer setup[cite: 1, 5]
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = RMSprop(params=self.params, lr=self.args.lr,
                                  alpha=self.args.optim_alpha, eps=self.args.optim_eps)

        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)
        self.hidden_states = None

    def init_hidden(self, batch_size=1):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def step(self, obs, avail_actions, t_env, test_mode=False):
        # Action selection logic is the same as in the other algorithms[cite: 1]
        agent_inputs = torch.tensor(obs, dtype=torch.float32, device=self.args.device)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        avail_actions_tensor = torch.tensor(avail_actions, dtype=torch.float32, device=self.args.device)
        chosen_actions = self.action_selector.select_action(
            agent_outs.unsqueeze(0), avail_actions_tensor.unsqueeze(0), t_env, test_mode=test_mode
        ).squeeze(0)

        return chosen_actions.cpu().numpy()

    def update(self, batch, t_env):
        """
        Compute QTRAN losses (TD loss, Opt loss, Nopt loss) and update[cite: 5]
        """
        if batch.device != self.args.device:
            batch.to(self.args.device)

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 1. Compute Q-values and hidden states over the whole episode with the current network[cite: 5]
        mac_out = []
        mac_hidden_states = []
        self.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, self.hidden_states = self.agent(batch["obs"][:, t], self.hidden_states)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.hidden_states)
        
        mac_out = torch.stack(mac_out, dim=1)  # (batch, time, n_agents, n_actions)
        mac_hidden_states = torch.stack(mac_hidden_states, dim=1) 
        # reshape to btav format[cite: 5]
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, batch.max_seq_length, self.n_agents, -1)

        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        # 2. Target network pass[cite: 5]
        target_mac_out = []
        target_mac_hidden_states = []
        target_hidden_states = self.target_agent.init_hidden().unsqueeze(0).expand(
            batch.batch_size, self.n_agents, -1
        )
        for t in range(batch.max_seq_length):
            target_agent_outs, target_hidden_states = self.target_agent(batch["obs"][:, t], target_hidden_states)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(target_hidden_states)
            
        target_mac_out = torch.stack(target_mac_out[:], dim=1)
        target_mac_hidden_states = torch.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, batch.max_seq_length, self.n_agents, -1)

        # 3. Masking and optimal action extraction[cite: 5]
        target_mac_out[avail_actions[:, :] == 0] = -9999999
        mac_out_maxs = mac_out.clone()
        mac_out_maxs[avail_actions == 0] = -9999999

        target_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
        max_actions_qvals, max_actions_current = mac_out_maxs[:, :].max(dim=3, keepdim=True)

        # 4. Compute QTRAN losses[cite: 5]
        # -- TD Loss --
        joint_qs, vs = self.mixer(batch[:, :-1], mac_hidden_states[:, :-1])

        if getattr(self.args, "double_q", True):
            max_actions_current_ = torch.zeros(size=(batch.batch_size, batch.max_seq_length, self.n_agents, self.n_actions), device=batch.device)
            max_actions_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
        else:
            max_actions = torch.zeros(size=(batch.batch_size, batch.max_seq_length, self.n_agents, self.n_actions), device=batch.device)
            max_actions_onehot = max_actions.scatter(3, target_max_actions[:, :], 1)
            
        target_joint_qs, target_vs = self.target_mixer(batch[:, 1:], hidden_states=target_mac_hidden_states[:, 1:], actions=max_actions_onehot[:, 1:])

        td_targets = rewards.reshape(-1, 1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_joint_qs
        td_error = (joint_qs - td_targets.detach())
        masked_td_error = td_error * mask.reshape(-1, 1)
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        # -- Opt Loss --
        if not getattr(self.args, "double_q", True):
            max_actions_current_ = torch.zeros(size=(batch.batch_size, batch.max_seq_length, self.n_agents, self.n_actions), device=batch.device)
            max_actions_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
            
        max_joint_qs, _ = self.mixer(batch[:, :-1], mac_hidden_states[:, :-1], actions=max_actions_onehot[:, :-1])
        
        opt_error = max_actions_qvals[:, :-1].sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
        masked_opt_error = opt_error * mask.reshape(-1, 1)
        opt_loss = (masked_opt_error ** 2).sum() / mask.sum()

        # -- Nopt Loss --
        nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1) - joint_qs.detach() + vs
        nopt_error = nopt_values.clamp(max=0)
        masked_nopt_error = nopt_error * mask.reshape(-1, 1)
        nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()

        # 5. Combine final loss and optimize[cite: 5]
        loss = td_loss + getattr(self.args, "opt_loss", 1.0) * opt_loss + getattr(self.args, "nopt_min_loss", 0.1) * nopt_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, getattr(self.args, "grad_norm_clip", 10))
        self.optimizer.step()

        return loss.item(), grad_norm.item()

    def update_targets(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, filename):
        save_dict = {
            'agent': self.agent.state_dict(),
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(save_dict, filename)
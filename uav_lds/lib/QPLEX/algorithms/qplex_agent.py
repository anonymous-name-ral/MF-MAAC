import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
import copy
from types import SimpleNamespace

# Import QPLEX core components
from modules.agents.rnn_agent import RNNAgent
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
from components.action_selectors import REGISTRY as action_REGISTRY

class QPLEXAgent(object):
    def __init__(self, config, obs_shape, n_actions, state_dim, n_agents):
        self.args = SimpleNamespace(**config)
        self.args.n_agents = n_agents
        self.args.n_actions = n_actions
        self.args.state_shape = [state_dim]
        
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        # 2. Agent and mixer initialization (parameter sharing)
        self.agent = RNNAgent(obs_shape, self.args).to(self.args.device)
        self.mixer = DMAQ_QattenMixer(self.args).to(self.args.device)
        
        # 3. Copy target networks (deepcopy after moving the originals to GPU, so targets are created on GPU too)
        self.target_agent = copy.deepcopy(self.agent)
        self.target_mixer = copy.deepcopy(self.mixer)
        
        # 4. Optimizer setup
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = RMSprop(params=self.params, lr=self.args.lr, 
                                alpha=self.args.optim_alpha, eps=self.args.optim_eps)
                                
        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)
        self.hidden_states = None

        # [2026-10-21] state_norm option (default False = original behavior): normalize mixer input state with running mean/std
        self.state_norm = getattr(self.args, "state_norm", False)
        if self.state_norm:
            self.sn_mean = torch.zeros(state_dim, device=self.args.device)
            self.sn_var = torch.ones(state_dim, device=self.args.device)
            self.sn_count = 1e-4
        self.last_q_stats = {}

    def init_hidden(self, batch_size=1):
        """
        Resets the RNN hidden state at the start of every new episode.
        """
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    @torch.no_grad()   # [2026-10-21] no gradients needed for action selection (same values, saves memory)
    def step(self, obs, avail_actions, t_env, test_mode=False):
        agent_inputs = torch.tensor(obs, dtype=torch.float32, device=self.args.device)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)


        
        avail_actions_tensor = torch.tensor(avail_actions, dtype=torch.float32, device=self.args.device)
        chosen_actions = self.action_selector.select_action(
            agent_outs.unsqueeze(0), avail_actions_tensor.unsqueeze(0), t_env, test_mode=test_mode
        ).squeeze(0)

        return chosen_actions.cpu().numpy()

    def update(self, batch, t_env):
        """
        Takes EpisodeBatch data, computes the QPLEX loss and updates the weights.
        """
        if batch.device != self.args.device:
            batch.to(self.args.device)
        if self.state_norm:
            self._normalize_batch_state(batch)

        # 1. Extract data from the batch (including masks)
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # 2. Compute Q-values over the whole episode with the current network (RNN unrolling)
        mac_out = []
        self.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, self.hidden_states = self.agent(batch["obs"][:, t], self.hidden_states)
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # (batch, time, n_agents, n_actions)

        # Extract Q-values of the actions actually taken
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Compute max Q-values for the advantage (masking unavailable actions)
        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        # 3. Target network pass ([2026-10-21] no_grad: target is detached so values are identical; only saves memory)
        target_mac_out = []
        target_hidden_states = self.target_agent.init_hidden().unsqueeze(0).expand(batch.batch_size, self.n_agents, -1)
        with torch.no_grad():
            for t in range(batch.max_seq_length):
                target_agent_outs, target_hidden_states = self.target_agent(batch["obs"][:, t], target_hidden_states)
                target_mac_out.append(target_agent_outs)
        target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # drop the first timestep

        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Apply Double Q-learning logic
        if getattr(self.args, "double_q", True):
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]

            cur_max_actions_onehot = torch.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).to(rewards.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_chosen_qvals = target_max_qvals # fallback

        # 4. Pass through the QPLEX mixer (value + advantage decomposition)
        # Current network mixing
        ans_chosen, q_attend_regs, head_entropies = self.mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True) #
        ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                   max_q_i=max_action_qvals, is_v=False) #
        chosen_action_qvals = ans_chosen + ans_adv #

        # Target network mixing ([2026-10-21] no_grad: same values, saves memory)
        with torch.no_grad():
            if getattr(self.args, "double_q", True): #
                target_chosen, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True) #
                target_adv, _, _ = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                     actions=cur_max_actions_onehot,
                                                     max_q_i=target_max_qvals, is_v=False)
                target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)

        # 5. TD-error and final loss computation
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals 
        td_error = (chosen_action_qvals - targets.detach()) #

        mask = mask.expand_as(td_error) #[
        masked_td_error = td_error * mask #

        # Add the QPLEX attention regularization term (q_attend_regs) to the L2 loss
        loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs #

        # 6. Backprop and optimization
        self.optimizer.zero_grad() #[
        loss.backward() #
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, getattr(self.args, "grad_norm_clip", 10)) #
        self.optimizer.step() #

        with torch.no_grad():   # [2026-10-21] diagnostic Q statistics (no effect on training)
            m = mask.sum()
            self.last_q_stats = {
                "Q/Q_tot_mean": ((chosen_action_qvals * mask).sum() / m).item(),
                "Q/Q_tot_absmax": (chosen_action_qvals.abs() * mask).max().item(),
                "Q/Target_mean": ((targets * mask).sum() / m).item(),
                "Q/Qi_mean": max_action_qvals.mean().item(),
                "Q/Qi_absmax": max_action_qvals.abs().max().item(),
                "Q/TD_abs_mean": (masked_td_error.abs().sum() / m).item(),
            }
        return loss.item(), grad_norm.item()

    def _normalize_batch_state(self, batch):
        """[2026-10-21] state_norm: update running mean/std, then replace batch state with a new normalized (±10 clip) tensor.
        Since a new tensor is used, the original state in the replay buffer is unchanged."""
        s = batch["state"]
        with torch.no_grad():
            x = s[batch["filled"].squeeze(-1).bool()]
            b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
            delta, tot = b_mean - self.sn_mean, self.sn_count + b_n
            self.sn_mean = self.sn_mean + delta * b_n / tot
            self.sn_var = (self.sn_var * self.sn_count + b_var * b_n + delta ** 2 * self.sn_count * b_n / tot) / tot
            self.sn_count = tot
            batch.data.transition_data["state"] = ((s - self.sn_mean) / torch.sqrt(self.sn_var + 1e-8)).clamp(-10, 10)

    def update_targets(self):
        """ Hard update of the target networks """
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, filename):
        save_dict = {
            'agent': self.agent.state_dict(),
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.state_norm:
            save_dict['state_norm'] = {'mean': self.sn_mean.cpu(), 'var': self.sn_var.cpu(), 'count': self.sn_count}
        torch.save(save_dict, filename)

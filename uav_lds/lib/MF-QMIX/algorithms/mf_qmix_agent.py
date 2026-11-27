import torch
import copy
from types import SimpleNamespace
from torch.optim import RMSprop

from modules.agents.mf_rnn_agent import MeanFieldRNNAgent
from modules.mixers.mf_qmix import CylindricalMFMixer
from components.action_selectors import REGISTRY as action_REGISTRY


class MFQMIXAgent(object):
    """
    Same skeleton as qmix_agent.py (QMIXAgent), but:
      - RNNAgent -> MeanFieldRNNAgent (also takes mean-field obs/action as input)
      - QMixer   -> CylindricalMFMixer (O(1) mixer based on moment embeddings)
      are swapped in.

    * Important: this class must receive "previous-timestep actions (one-hot)" and "group mean-field obs/actions"
      on every step()/update() call. The training loop in main_mfqmix.py
      must compute these values and pass them in; for an example of that computation, see the
      additional logic (relative to main_qmix.py) described separately.
    """

    def __init__(self, config, obs_shape, n_actions, state_dim, n_agents):
        self.args = SimpleNamespace(**config)
        self.args.n_agents = n_agents
        self.args.n_actions = n_actions
        self.args.state_shape = [state_dim]

        # Dimension of the mean-field state (mu_s) = one agent's obs dim (fixed size, independent of agent count)
        self.args.mean_field_state_dim = obs_shape

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_shape = obs_shape

        # 1. Mean-Field Agent Network
        self.agent = MeanFieldRNNAgent(obs_shape, self.args).to(self.args.device)

        # 2. Cylindrical Mixing Network
        self.mixer = CylindricalMFMixer(self.args).to(self.args.device)

        # 3. Target networks
        self.target_agent = copy.deepcopy(self.agent)
        self.target_mixer = copy.deepcopy(self.mixer)

        # 4. Optimizer
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = RMSprop(params=self.params, lr=self.args.lr,
                                  alpha=self.args.optim_alpha, eps=self.args.optim_eps)

        self.action_selector = action_REGISTRY[self.args.action_selector](self.args)
        self.hidden_states = None

        # Diagnostic options (2026-10-20, defaults = original behavior)
        #   state_norm: normalize the mixer input (mean-field state) with running mean/std
        #   seq_chunk : in update(), detach GRU hidden every seq_chunk steps (truncated BPTT, 0 = full 1000 steps)
        self.state_norm = getattr(self.args, "state_norm", False)
        self.seq_chunk = getattr(self.args, "seq_chunk", 0)
        if self.state_norm:
            self.sn_mean = torch.zeros(obs_shape, device=self.args.device)
            self.sn_var = torch.ones(obs_shape, device=self.args.device)
            self.sn_count = 1e-4

    def init_hidden(self, batch_size=1):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)


    def _compute_mean_field(self, obs, last_action_onehot):
        """
        obs                : (bs, n_agents, obs_dim)
        last_action_onehot : (bs, n_agents, n_actions)

        For each agent i, computes "the mean over all other agents, excluding itself".
        """
        n = self.n_agents
        if n > 1:
            obs_sum = obs.sum(dim=1, keepdim=True)                       # (bs, 1, obs_dim)
            mean_field_obs = (obs_sum - obs) / (n - 1)                    # (bs, n_agents, obs_dim) — differs per agent

            act_sum = last_action_onehot.sum(dim=1, keepdim=True)         # (bs, 1, n_actions)
            mean_field_action = (act_sum - last_action_onehot) / (n - 1)  # (bs, n_agents, n_actions)
        else:
            # With only one agent there are no "neighbors", so use 0
            mean_field_obs = torch.zeros_like(obs)
            mean_field_action = torch.zeros_like(last_action_onehot)

        return mean_field_obs, mean_field_action
    

    @torch.no_grad()   # no gradients needed for action selection (same values, saves memory)
    def step(self, obs, last_action_onehot, avail_actions, t_env, test_mode=False):
        """
        obs                : (n_agents, obs_dim) numpy
        last_action_onehot : (n_agents, n_actions) numpy — one-hot of each agent's action at the previous step
                              (all zero vectors on the first step of an episode)
        avail_actions       : (n_agents, n_actions)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.args.device).unsqueeze(0)          # 
        last_act_t = torch.tensor(last_action_onehot, dtype=torch.float32, device=self.args.device).unsqueeze(0)
        # ******************* Added: build Agent ID one-hot (1, n_agents, n_agents)
        agent_id_onehot = torch.eye(self.n_agents, device=self.args.device).unsqueeze(0)

        mean_field_obs, mean_field_action = self._compute_mean_field(obs_t, last_act_t)

        # ****************** Changed: pass agent_id_onehot
        agent_outs, self.hidden_states = self.agent(
            obs_t, last_act_t, mean_field_obs, mean_field_action, agent_id_onehot, self.hidden_states
        )

        avail_actions_tensor = torch.tensor(avail_actions, dtype=torch.float32, device=self.args.device).unsqueeze(0)
        chosen_actions = self.action_selector.select_action(
            agent_outs, avail_actions_tensor, t_env, test_mode=test_mode
        ).squeeze(0)

        return chosen_actions.cpu().numpy()

    def update(self, batch, t_env):
        if batch.device != self.args.device:
            batch.to(self.args.device)

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"]          # (bs, T, n_agents, n_actions) — already present via OneHot preprocessing
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        obs = batch["obs"]                                  # (bs, T, n_agents, obs_dim)
        state = batch["state"]                               # (bs, T, state_dim) = (bs, T, obs_dim * n_agents)

        bs, T = obs.shape[0], obs.shape[1]

        # **************Added: build Agent ID one-hot for the update loop (bs, n_agents, n_agents)
        agent_id_onehot = torch.eye(self.n_agents, device=self.args.device).unsqueeze(0).expand(bs, -1, -1)

        # Build previous-timestep action (one-hot) sequence: zero vector at t=0, actions_onehot[:, t-1] for t>=1
        zero_action = torch.zeros_like(actions_onehot[:, 0:1])
        last_actions_onehot = torch.cat([zero_action, actions_onehot[:, :-1]], dim=1)  # (bs, T, n_agents, n_actions)

        # 1. Compute Q-values over the whole episode with the current network
        mac_out = []
        self.init_hidden(bs)
        for t in range(T):
            if self.seq_chunk and t > 0 and t % self.seq_chunk == 0:
                self.hidden_states = self.hidden_states.detach()
            mean_field_obs, mean_field_action = self._compute_mean_field(obs[:, t], last_actions_onehot[:, t])
            # ******************** Changed: pass agent_id_onehot
            agent_outs, self.hidden_states = self.agent(
                obs[:, t], last_actions_onehot[:, t], mean_field_obs, mean_field_action, agent_id_onehot, self.hidden_states
            )
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # (bs, T, n_agents, n_actions)

        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # 2. Target network pass (no_grad: target is detached so values are identical; only saves 1000-step graph memory)
        target_mac_out = []
        target_hidden_states = self.target_agent.init_hidden().unsqueeze(0).expand(bs, self.n_agents, -1)

        with torch.no_grad():
            for t in range(T):
                mean_field_obs, mean_field_action = self._compute_mean_field(obs[:, t], last_actions_onehot[:, t])

                # *************Changed: also pass agent_id_onehot when calling target_agent
                target_agent_outs, target_hidden_states = self.target_agent(
                    obs[:, t], last_actions_onehot[:, t], mean_field_obs, mean_field_action, agent_id_onehot, target_hidden_states
                )

                target_mac_out.append(target_agent_outs)
        target_mac_out = torch.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if getattr(self.args, "double_q", True):
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # 3. Compute mean-field state mu_s: reshape state to (bs, T, n_agents, obs_dim), then average over agent dim
        state_reshaped = state.view(bs, T, self.n_agents, self.obs_shape)
        mean_field_state = state_reshaped.mean(dim=2)  # (bs, T, obs_dim) — fixed size independent of agent count
        if self.state_norm:
            mean_field_state = self._normalize_state(mean_field_state, batch["filled"])

        #=============================== wandb logging ======================================#
        #qi_raw = chosen_action_qvals.clone().detach()   # (bs, T-1, n_agents) 
        # ===================================== =============== ======================================== #

        chosen_action_qvals_flat = chosen_action_qvals.reshape(-1, self.n_agents)
        mean_field_state_cur = mean_field_state[:, :-1].reshape(-1, self.obs_shape)

        #=============================== wandb logging ======================================#
        #with torch.no_grad():   # also extract moment values separately for logging
        #    moments = self.mixer.compute_moments(chosen_action_qvals_flat)
        # ===================================== =============== ======================================== #



        chosen_action_qvals = self.mixer(chosen_action_qvals_flat, mean_field_state_cur).view(bs, T - 1, 1)

        target_max_qvals_flat = target_max_qvals.reshape(-1, self.n_agents)
        mean_field_state_next = mean_field_state[:, 1:].reshape(-1, self.obs_shape)
        with torch.no_grad():
            target_max_qvals = self.target_mixer(target_max_qvals_flat, mean_field_state_next).view(bs, T - 1, 1)

        # 5. TD-error and loss
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, getattr(self.args, "grad_norm_clip", 10))
        self.optimizer.step()


        # ===================================== wandb logging ======================================== #
        '''
        with torch.no_grad():
            stats = {
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "td_error_abs_mean": masked_td_error.abs().sum().item() / mask.sum().item(),
                "q_tot_mean": chosen_action_qvals.mean().item(),
                "q_tot_max": chosen_action_qvals.max().item(),
                "target_q_mean": targets.mean().item(),
                "qi_mean": qi_raw.mean().item(),
                "qi_std": qi_raw.std().item(),
                "qi_min": qi_raw.min().item(),
            }
            # Per-moment values (K=1: mu1 only, K=2: both mu1/mu2)
            for k in range(moments.shape[1]):
                stats[f"moment_{k+1}_mean"] = moments[:, k].mean().item()

            # Hypernet weight norms
            stats["hyper_w1_norm"] = self.mixer.hyper_w1[-1].weight.norm().item() if isinstance(self.mixer.hyper_w1, torch.nn.Sequential) else self.mixer.hyper_w1.weight.norm().item()
            stats["hyper_w_final_norm"] = self.mixer.hyper_w_final[-1].weight.norm().item() if isinstance(self.mixer.hyper_w_final, torch.nn.Sequential) else self.mixer.hyper_w_final.weight.norm().item()
        # ===================================== =============== ======================================== #
        '''
        with torch.no_grad():
            n = mask.sum()
            qi = batch["filled"][:, :-1].float()
            q_stats = {
                "Q/Q_tot_mean": ((chosen_action_qvals * mask).sum() / n).item(),
                "Q/Q_tot_absmax": (chosen_action_qvals.abs() * mask).max().item(),
                "Q/Target_mean": ((targets * mask).sum() / n).item(),
                "Q/Qi_mean": ((chosen_action_qvals_flat.view(bs, T - 1, self.n_agents) * qi).sum() / (qi.sum() * self.n_agents)).item(),
                "Q/Qi_absmax": chosen_action_qvals_flat.abs().max().item(),
                "Q/TD_abs_mean": (masked_td_error.abs().sum() / n).item(),
            }
        return loss.item(), grad_norm.item(), q_stats

    def _normalize_state(self, s, filled):
        """s: (bs, T, obs_dim), filled: (bs, T, 1). Update running mean/std, then normalize (±10 clip)."""
        with torch.no_grad():
            x = s[filled.squeeze(-1).bool()]
            b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
            delta, tot = b_mean - self.sn_mean, self.sn_count + b_n
            self.sn_mean = self.sn_mean + delta * b_n / tot
            self.sn_var = (self.sn_var * self.sn_count + b_var * b_n + delta ** 2 * self.sn_count * b_n / tot) / tot
            self.sn_count = tot
        return ((s - self.sn_mean) / torch.sqrt(self.sn_var + 1e-8)).clamp(-10, 10)

    def update_targets(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_update_targets(self, tau):
        with torch.no_grad():
            for net, tgt in ((self.agent, self.target_agent), (self.mixer, self.target_mixer)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(1 - tau).add_(p, alpha=tau)

    def save(self, filename):
        save_dict = {
            'agent': self.agent.state_dict(),
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.state_norm:
            save_dict['state_norm'] = {'mean': self.sn_mean.cpu(), 'var': self.sn_var.cpu(), 'count': self.sn_count}
        torch.save(save_dict, filename)
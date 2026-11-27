"""
Phase 0 replay buffer + actor/critic input builders.

Transitions are stored once (no per-transition window copies). Observation
windows (last k frames + previous actions) are gathered at sample time from
`t_in_ep` / `ep_id`; frames before the episode start are zero-masked.
The same input builder is used for rollouts, evaluation and sampled batches.
"""
import numpy as np
import torch


class InputSpec(object):
    def __init__(self, obs_dim, act_dim, nagents, state_dim, actor_stack_k=1,
                 actor_prev_action=False, actor_mf_input=False,
                 actor_state_input=False, critic_stack_k=1,
                 critic_state_input=False, critic_prev_action=False,
                 actor_agent_id=False, critic_agent_id=False):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.nagents = nagents
        self.state_dim = state_dim
        self.actor_stack_k = int(actor_stack_k)
        self.actor_prev_action = bool(actor_prev_action)
        self.actor_mf_input = bool(actor_mf_input)
        self.actor_state_input = bool(actor_state_input)
        self.critic_stack_k = int(critic_stack_k)
        self.critic_state_input = bool(critic_state_input)
        self.critic_prev_action = bool(critic_prev_action)   # E1: previous action of each critic frame
        self.actor_agent_id = bool(actor_agent_id)           # E3: one-hot agent id (shared networks)
        self.critic_agent_id = bool(critic_agent_id)
        # number of frames that must be kept / gathered
        self.k = max(self.actor_stack_k, self.critic_stack_k)

    @property
    def needs_state(self):
        return self.actor_state_input or self.critic_state_input

    @property
    def actor_dim(self):
        d = self.actor_stack_k * self.obs_dim
        if self.actor_prev_action:
            d += self.actor_stack_k * self.act_dim
        if self.actor_mf_input:
            d += self.obs_dim + self.act_dim
        if self.actor_state_input:
            d += self.state_dim
        if self.actor_agent_id:
            d += self.nagents
        return d

    @property
    def critic_obs_dim(self):
        d = self.critic_stack_k * self.obs_dim
        if self.critic_prev_action:
            d += self.critic_stack_k * self.act_dim
        if self.critic_agent_id:
            d += self.nagents
        return d

    @property
    def ego_dim(self):
        """Input dim of the critic ego (state) encoder; None = unchanged."""
        if not self.critic_state_input:
            return None
        return self.critic_obs_dim + self.state_dim

    def build(self, obs_win, act_win, state):
        """
        obs_win: (B, k, n, od) oldest -> newest, zero where masked
        act_win: (B, k, n, ad) previous action of each frame, zero where masked
        state:   (B, sd) or None
        Returns (actor_in, critic_obs, ego) as lists (per agent) of float32 arrays.
        """
        B, k, n, od = obs_win.shape
        ka, kc = self.actor_stack_k, self.critic_stack_k
        cur_obs = obs_win[:, -1]          # (B, n, od)
        cur_pact = act_win[:, -1]         # (B, n, ad)
        if self.actor_mf_input:
            sum_obs = cur_obs.sum(axis=1)
            sum_pact = cur_pact.sum(axis=1)
        actor_in, critic_obs, ego = [], [], []
        for i in range(n):
            parts = [obs_win[:, k - ka:, i].reshape(B, ka * od)]
            if self.actor_prev_action:
                parts.append(act_win[:, k - ka:, i].reshape(B, ka * self.act_dim))
            if self.actor_mf_input:
                denom = max(n - 1, 1)
                parts.append((sum_obs - cur_obs[:, i]) / denom)
                parts.append((sum_pact - cur_pact[:, i]) / denom)
            if self.actor_state_input:
                parts.append(state)
            if self.actor_agent_id or self.critic_agent_id:
                aid = np.zeros((B, n), np.float32)
                aid[:, i] = 1.0
            if self.actor_agent_id:
                parts.append(aid)
            actor_in.append(np.concatenate(parts, axis=1).astype(np.float32))
            cparts = [obs_win[:, k - kc:, i].reshape(B, kc * od)]
            if self.critic_prev_action:
                cparts.append(act_win[:, k - kc:, i].reshape(B, kc * self.act_dim))
            if self.critic_agent_id:
                cparts.append(aid)
            c = np.concatenate(cparts, axis=1).astype(np.float32)
            critic_obs.append(c)
            if self.critic_state_input:
                ego.append(np.concatenate([c, state], axis=1).astype(np.float32))
        return actor_in, critic_obs, (ego if self.critic_state_input else None)


class RolloutHistory(object):
    """Per-env last-k frames for rollouts / evaluation (reset each episode)."""
    def __init__(self, spec, n_envs):
        self.spec = spec
        self.obs = np.zeros((n_envs, spec.k, spec.nagents, spec.obs_dim), np.float32)
        self.act = np.zeros((n_envs, spec.k, spec.nagents, spec.act_dim), np.float32)
        self.state = None

    def reset(self, e, obs0, state0):
        self.obs[e] = 0.0
        self.act[e] = 0.0
        self.obs[e, -1] = obs0
        self._set_state(e, state0)

    def push(self, e, next_obs, action_onehot, next_state):
        self.obs[e, :-1] = self.obs[e, 1:]
        self.act[e, :-1] = self.act[e, 1:]
        self.obs[e, -1] = next_obs
        self.act[e, -1] = action_onehot   # previous action of the new frame
        self._set_state(e, next_state)

    def _set_state(self, e, s):
        if not self.spec.needs_state:
            return
        if self.state is None:
            self.state = np.zeros((self.obs.shape[0], self.spec.state_dim), np.float32)
        self.state[e] = s

    def inputs(self):
        return self.spec.build(self.obs, self.act, self.state)


class Phase0ReplayBuffer(object):
    def __init__(self, max_steps, spec, dtype=np.float16):
        self.N = int(max_steps)
        self.spec = spec
        n, od, ad = spec.nagents, spec.obs_dim, spec.act_dim
        self.obs = np.zeros((self.N, n, od), dtype)
        self.next_obs = np.zeros((self.N, n, od), dtype)
        self.acs = np.zeros((self.N, n, ad), np.uint8)
        self.rews = np.zeros((self.N, n), np.float32)
        self.dones = np.zeros(self.N, np.uint8)      # real terminals only
        self.t_in_ep = np.zeros(self.N, np.int32)
        self.ep_id = np.full(self.N, -1, np.int64)
        if spec.needs_state:
            self.state = np.zeros((self.N, spec.state_dim), dtype)
            self.next_state = np.zeros((self.N, spec.state_dim), dtype)
        self.filled_i = 0
        self.curr_i = 0

    def __len__(self):
        return self.filled_i

    def push(self, obs, acs, rews, next_obs, dones, t_in_ep, ep_id,
             state=None, next_state=None):
        """All arguments batched over envs: obs (E, n, od), acs (E, n, ad), ..."""
        E = obs.shape[0]
        idx = (self.curr_i + np.arange(E)) % self.N
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.acs[idx] = acs
        self.rews[idx] = rews
        self.dones[idx] = dones
        self.t_in_ep[idx] = t_in_ep
        self.ep_id[idx] = ep_id
        if self.spec.needs_state:
            self.state[idx] = state
            self.next_state[idx] = next_state
        self.curr_i = (self.curr_i + E) % self.N
        self.filled_i = min(self.filled_i + E, self.N)

    def _window(self, inds, nxt):
        """Gather (B, k, n, od) obs window and (B, k, n, ad) prev-action window."""
        k = self.spec.k
        B = len(inds)
        n, od, ad = self.spec.nagents, self.spec.obs_dim, self.spec.act_dim
        t = self.t_in_ep[inds]
        ep = self.ep_id[inds]
        obs_w = np.zeros((B, k, n, od), np.float32)
        act_w = np.zeros((B, k, n, ad), np.float32)
        shift = 1 if nxt else 0          # window of time t+1 instead of t
        for j in range(k):               # j = steps back from the newest frame
            slot = k - 1 - j
            if nxt and j == 0:
                obs_w[:, slot] = self.next_obs[inds]
            else:
                src = (inds - j + shift) % self.N
                valid = (t + shift - j >= 0) & (self.ep_id[src] == ep)
                obs_w[valid, slot] = self.obs[src[valid]]
            # previous action of the frame at time (t + shift - j)
            psrc = (inds - j - 1 + shift) % self.N
            pvalid = (t + shift - j - 1 >= 0) & (self.ep_id[psrc] == ep)
            act_w[pvalid, slot] = self.acs[psrc[pvalid]]
        return obs_w, act_w

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)
        dev = torch.device('cuda') if to_gpu else torch.device('cpu')
        cast = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
        spec = self.spec

        st = nst = None
        if spec.needs_state:
            st = self.state[inds].astype(np.float32)
            nst = self.next_state[inds].astype(np.float32)
        ow, aw = self._window(inds, nxt=False)
        now, naw = self._window(inds, nxt=True)
        a_in, c_obs, ego = spec.build(ow, aw, st)
        na_in, nc_obs, nego = spec.build(now, naw, nst)

        n = spec.nagents
        if norm_rews:
            rf = self.rews[:self.filled_i]
            mean, std = rf.mean(axis=0), rf.std(axis=0)
            rews = [(self.rews[inds, i] - mean[i]) / (std[i] + 1e-7) for i in range(n)]
        else:
            rews = [self.rews[inds, i] for i in range(n)]
        dones = self.dones[inds].astype(np.float32)
        to_list = lambda xs: None if xs is None else [cast(x) for x in xs]
        return {
            'actor_in': to_list(a_in), 'next_actor_in': to_list(na_in),
            'critic_obs': to_list(c_obs), 'next_critic_obs': to_list(nc_obs),
            'ego': to_list(ego), 'next_ego': to_list(nego),
            'acs': [cast(self.acs[inds, i]) for i in range(n)],
            'rews': [cast(r) for r in rews],
            'dones': [cast(dones) for _ in range(n)],
        }

    def nbytes(self):
        return sum(v.nbytes for v in self.__dict__.values() if isinstance(v, np.ndarray))

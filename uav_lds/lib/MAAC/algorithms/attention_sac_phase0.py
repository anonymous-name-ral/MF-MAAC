"""
Phase 0 variant of AttentionSAC (attention_sac_db.py).

Differences from attention_sac_db.AttentionSAC:
- critic class chosen by `critic_type` (mf_p / msf_p / plain)
- actor input and critic input are decoupled (windows, MF/state features)
- optional ego-encoder input (critic_state_input) via `ego_states`
- metrics are returned instead of logged per update; niter += 1 per update
The update rules (soft Q target, MAAC counterfactual baseline, regularisers,
grad clipping, shared-grad scaling) are unchanged.
"""
import importlib
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent

MSELoss = torch.nn.MSELoss()

# critic_type -> (module, model name). MF-P encoder h->2h->h, MSF-P encoder 2h->2h->h.
CRITICS = {
    'plain':       ('utils.critics_maac',          'MAAC'),
    'mf_p':        ('utils.critics_mf_maac_p',     'MF-MAAC-P'),
    'mf_t':        ('utils.critics_mf_maac_t',     'MF-MAAC-T'),
    'msf_p':       ('utils.critics_msf_maac_p',    'MSF-MAAC-P'),
    'msf_t':       ('utils.critics_msf_maac_t',    'MSF-MAAC-T'),
    'mahac':       ('utils.critics_mahac',         'MAHAC'),
    'mahac_mf_p':  ('utils.critics_mf_mahac_p',    'MF-MAHAC-P'),
    'mahac_mf_t':  ('utils.critics_mf_mahac_t',    'MF-MAHAC-T'),
    'mahac_msf_p': ('utils.critics_msf_mahac_p',   'MSF-MAHAC-P'),
    'mahac_msf_t': ('utils.critics_msf_mahac_t',   'MSF-MAHAC-T'),
    'fmaac':       ('utils.critics_f_maac',        'F-MAAC'),
    'fmaac_mf_p':  ('utils.critics_mf_f_maac_p',   'MF-F-MAAC-P'),
    'fmaac_mf_t':  ('utils.critics_mf_f_maac_t',   'MF-F-MAAC-T'),
    'fmaac_msf_p': ('utils.critics_msf_f_maac_p',  'MSF-F-MAAC-P'),
    'fmaac_msf_t': ('utils.critics_msf_f_maac_t',  'MSF-F-MAAC-T'),
}
CRITIC_MODULES = {k: v[0] for k, v in CRITICS.items()}
MODEL_NAMES = {k: v[1] for k, v in CRITICS.items()}


class PartialBN(nn.Module):
    """BatchNorm1d(affine=False) over every input dim except a one-hot agent id at [start, start+n).
    A per-agent batch has a constant id, so normalising it erases the id in training and blows it up
    (running var ~ 0) in eval mode; the id is passed through unchanged instead."""
    def __init__(self, dim, start, n):
        super(PartialBN, self).__init__()
        self.start, self.n = start, n
        self.bn = nn.BatchNorm1d(dim - n, affine=False)

    def forward(self, x):
        s, e = self.start, self.start + self.n
        y = self.bn(torch.cat([x[:, :s], x[:, e:]], dim=1))
        return torch.cat([y[:, :s], x[:, s:e], y[:, s:]], dim=1)


def patch_actor_id_bn(policy, n):
    """actor input = [..., agent id (last n dims)]"""
    if isinstance(policy.in_fn, nn.BatchNorm1d):
        d = policy.fc1.in_features
        policy.in_fn = PartialBN(d, d - n, n)


def patch_critic_id_bn(critic, sdim, n):
    """critic encoder input = [critic obs (agent id = last n of sdim), action]; ego encoder input = critic obs"""
    for enc in critic.critic_encoders:
        if hasattr(enc, 'enc_bn'):
            enc.enc_bn = PartialBN(enc.enc_fc1.in_features, sdim - n, n)
    for se in critic.state_encoders:
        if hasattr(se, 's_enc_bn'):
            se.s_enc_bn = PartialBN(se.s_enc_fc1.in_features, sdim - n, n)


class FrameGRU(nn.Module):
    """E4: encode a k-step history with a GRU instead of concatenating it.
    Input row layout (as built by InputSpec): [obs_{t-k+1..t} (k*od) | prev_act_{t-k+1..t} (k*ad) | extra].
    Frames [obs_i, prev_act_i] (od+ad) are normalised per frame (BatchNorm, affine=False, like the concat input BN),
    fed oldest -> newest into a 1-layer GRU; the last hidden state (hid) is returned, extra dims passed through."""
    def __init__(self, k, od, ad, hid=128):
        super(FrameGRU, self).__init__()
        self.k, self.od, self.ad, self.hid = k, od, ad, hid
        self.bn = nn.BatchNorm1d(od + ad, affine=False)
        self.gru = nn.GRU(od + ad, hid, num_layers=1, batch_first=True)

    def forward(self, x):
        B, k, od, ad = x.shape[0], self.k, self.od, self.ad
        frames = torch.cat([x[:, :k * od].reshape(B, k, od), x[:, k * od:k * (od + ad)].reshape(B, k, ad)], dim=2)
        frames = self.bn(frames.reshape(B * k, od + ad)).reshape(B, k, od + ad)
        _, h = self.gru(frames)
        out = h[-1]
        extra = x[:, k * (od + ad):]
        return torch.cat([out, extra], dim=1) if extra.shape[1] else out


class SeqPolicy(nn.Module):
    """FrameGRU + DiscretePolicy(hid + extra) (same policy layers as the concat actor)."""
    def __init__(self, k, od, ad, extra, out_dim, hidden_dim, hid=128):
        super(SeqPolicy, self).__init__()
        from utils.policies import DiscretePolicy
        self.enc = FrameGRU(k, od, ad, hid)
        self.pol = DiscretePolicy(hid + extra, out_dim, hidden_dim=hidden_dim)

    def forward(self, obs, **kw):
        return self.pol(self.enc(obs), **kw)


def make_seq_agent(agent, k, od, ad, extra, out_dim, hidden_dim, lr):
    agent.policy = SeqPolicy(k, od, ad, extra, out_dim, hidden_dim)
    agent.target_policy = SeqPolicy(k, od, ad, extra, out_dim, hidden_dim)
    hard_update(agent.target_policy, agent.policy)
    agent.policy_optimizer = Adam(agent.policy.parameters(), lr=lr)
    return agent


class SeqCritic(nn.Module):
    """E4: per-agent FrameGRU in front of an unchanged attention critic. The GRU output h_j replaces the concatenated
    history everywhere the critic used it (ego/state encoder and the (obs, action) encoder seen by attention/MF)."""
    def __init__(self, critic_cls, nagents, k, od, ad, act_dim, hid=128, **kw):
        super(SeqCritic, self).__init__()
        self.enc = nn.ModuleList([FrameGRU(k, od, ad, hid) for _ in range(nagents)])
        self.inner = critic_cls([(hid, act_dim)] * nagents, **kw)

    def forward(self, inps, **kw):
        return self.inner([(enc(s), a) for enc, (s, a) in zip(self.enc, inps)], **kw)

    def scale_shared_grads(self):
        self.inner.scale_shared_grads()


def get_critic_cls(critic_type):
    return importlib.import_module(CRITIC_MODULES[critic_type]).AttentionCritic


class AttentionSACPhase0(object):
    def __init__(self, agent_init_params, sa_size, critic_type='mf_p', ego_dim=None,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01, reward_scale=10.,
                 pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                 share_actor=False, share_critic=False, actor_id_dim=0, critic_id_dim=0,
                 critic_encoder='concat', actor_encoder='concat', seq=None, **kwargs):
        self.nagents = len(sa_size)
        # E3: parameter sharing (inputs carry a one-hot agent id, see InputSpec)
        self.share_actor, self.share_critic = bool(share_actor), bool(share_critic)
        if self.share_actor:
            shared = AttentionAgent(lr=pi_lr, hidden_dim=pol_hidden_dim, **agent_init_params[0])
            self.agents = [shared] * self.nagents
        else:
            self.agents = [AttentionAgent(lr=pi_lr, hidden_dim=pol_hidden_dim, **params)
                           for params in agent_init_params]
        if actor_encoder == 'gru':                 # E4b: GRU over the actor's k-step history
            for a in self._unique_agents():
                make_seq_agent(a, seq['k_actor'], seq['od'], seq['ad'], seq['actor_extra'],
                               agent_init_params[0]['num_out_pol'], pol_hidden_dim, pi_lr)
        if actor_id_dim:
            for a in self._unique_agents():
                patch_actor_id_bn(a.policy, actor_id_dim)
                patch_actor_id_bn(a.target_policy, actor_id_dim)
                a.policy_optimizer = Adam(a.policy.parameters(), lr=pi_lr)
        critic_cls = get_critic_cls(critic_type)
        if critic_encoder == 'gru':                # E4a: GRU over the critic's k-step history
            assert ego_dim is None and not share_critic
            mk = lambda: SeqCritic(critic_cls, len(sa_size), seq['k_critic'], seq['od'], seq['ad'], sa_size[0][1],
                                   hidden_dim=critic_hidden_dim, attend_heads=attend_heads, ego_dim=None)
            self.critic, self.target_critic = mk(), mk()
        else:
            self.critic = critic_cls(sa_size, hidden_dim=critic_hidden_dim,
                                     attend_heads=attend_heads, ego_dim=ego_dim)
            self.target_critic = critic_cls(sa_size, hidden_dim=critic_hidden_dim,
                                            attend_heads=attend_heads, ego_dim=ego_dim)
        if self.share_critic:
            for c in (self.critic, self.target_critic):
                self._share_per_agent_modules(c)
        if critic_id_dim:
            for c in (self.critic, self.target_critic):
                patch_critic_id_bn(c, sa_size[0][0], critic_id_dim)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'
        self.niter = 0

    def _share_per_agent_modules(self, critic):
        """One encoder / ego encoder / Q head for all agents (attention + MF modules are shared already)."""
        for name in ('critic_encoders', 'state_encoders', 'critics'):
            ml = getattr(critic, name)
            for i in range(1, len(ml)):
                ml[i] = ml[0]

    def _unique_agents(self):
        return self.agents[:1] if self.share_actor else self.agents

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, actor_inputs, explore=False):
        return [a.step(x, explore=explore) for a, x in zip(self.agents, actor_inputs)]

    def update_critic(self, s, soft=True):
        next_acs, next_log_pis = [], []
        for pi, ob in zip(self.target_policies, s['next_actor_in']):
            a, lp = pi(ob, return_log_pi=True)
            next_acs.append(a)
            next_log_pis.append(lp)
        next_qs = self.target_critic(list(zip(s['next_critic_obs'], next_acs)),
                                     ego_states=s['next_ego'])
        critic_rets = self.critic(list(zip(s['critic_obs'], s['acs'])), regularize=True,
                                  ego_states=s['ego'])
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (s['rews'][a_i].view(-1, 1) +
                        self.gamma * nq * (1 - s['dones'][a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg
        q_loss.backward()
        self.critic.scale_shared_grads()
        if self.share_critic:   # the shared ego encoder / Q head collect every agent's gradient
            for m in (self.critic.state_encoders[0], self.critic.critics[0]):
                for p in m.parameters():
                    if p.grad is not None:
                        p.grad.mul_(1. / self.nagents)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        self.niter += 1
        return {'q_loss': q_loss.item(), 'q_grad_norm': float(grad_norm)}

    def update_policies(self, s, soft=True):
        samp_acs, all_probs, all_log_pis, all_pol_regs, ents = [], [], [], [], []
        for pi, ob in zip(self.policies, s['actor_in']):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)
            ents.append(ent.item())
        critic_rets = self.critic(list(zip(s['critic_obs'], samp_acs)), return_all_q=True,
                                  ego_states=s['ego'])
        total = 0.0
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(
                range(self.nagents), all_probs, all_log_pis, all_pol_regs, critic_rets):
            agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg
            disable_gradients(self.critic)
            # shared actor: average the agents' losses, one optimizer step per update
            (pol_loss / self.nagents if self.share_actor else pol_loss).backward()
            enable_gradients(self.critic)
            if not self.share_actor:
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
                agent.policy_optimizer.step()
                agent.policy_optimizer.zero_grad()
            total += pol_loss.item()
        if self.share_actor:
            agent = self.agents[0]
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            agent.policy_optimizer.step()
            agent.policy_optimizer.zero_grad()
        return {'pol_loss': total / self.nagents, 'entropy': sum(ents) / self.nagents}

    def update_all_targets(self):
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self._unique_agents():
            soft_update(a.target_policy, a.policy, self.tau)

    def _move(self, device):
        fn = (lambda x: x.cuda()) if device == 'gpu' else (lambda x: x.cpu())
        return fn

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        fn = self._move(device)
        if self.pol_dev != device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if self.critic_dev != device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if self.trgt_pol_dev != device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if self.trgt_critic_dev != device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        fn = self._move(device)
        if self.pol_dev != device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename, extra=None):
        self.prep_training(device='cpu')
        save_dict = {'init_dict': self.init_dict,
                     'phase0': extra,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_spec(cls, spec, critic_type='mf_p', gamma=0.95, tau=0.01, pi_lr=0.01,
                       q_lr=0.01, reward_scale=10., pol_hidden_dim=128,
                       critic_hidden_dim=128, attend_heads=4, share_actor=False, share_critic=False,
                       critic_encoder='concat', actor_encoder='concat'):
        agent_init_params = [{'num_in_pol': spec.actor_dim, 'num_out_pol': spec.act_dim}
                             for _ in range(spec.nagents)]
        sa_size = [(spec.critic_obs_dim, spec.act_dim) for _ in range(spec.nagents)]
        init_dict = {'gamma': gamma, 'tau': tau, 'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale, 'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim, 'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params, 'sa_size': sa_size,
                     'critic_type': critic_type, 'ego_dim': spec.ego_dim,
                     'share_actor': share_actor, 'share_critic': share_critic,
                     'actor_id_dim': spec.nagents if spec.actor_agent_id else 0,
                     'critic_id_dim': spec.nagents if spec.critic_agent_id else 0,
                     'critic_encoder': critic_encoder, 'actor_encoder': actor_encoder,
                     'seq': {'k_critic': spec.critic_stack_k, 'k_actor': spec.actor_stack_k, 'od': spec.obs_dim,
                             'ad': spec.act_dim,
                             'actor_extra': (spec.obs_dim + spec.act_dim) if spec.actor_mf_input else 0}}
        if critic_encoder == 'gru':
            assert spec.critic_prev_action and spec.critic_stack_k > 1 and not spec.critic_agent_id
        if actor_encoder == 'gru':
            assert spec.actor_prev_action and spec.actor_stack_k > 1 and not (spec.actor_agent_id or spec.actor_state_input)
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        save_dict = torch.load(filename, map_location='cpu')
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        if load_critic:
            cp = save_dict['critic_params']
            instance.critic.load_state_dict(cp['critic'])
            instance.target_critic.load_state_dict(cp['target_critic'])
            instance.critic_optimizer.load_state_dict(cp['critic_optimizer'])
        return instance, save_dict.get('phase0')

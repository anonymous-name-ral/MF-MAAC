import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
import warnings


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions + Global MSF Token.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1,
                 ego_dim=None):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        # [MSF-Token change] Takes Mean(hidden_dim) + Std(hidden_dim), so input is 2 * hidden_dim
        # Output dim stays hidden_dim to remain compatible with the Key/Value extractors.
        self.global_token_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim, hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            edim = sdim if ego_dim is None else ego_dim
            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(edim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(edim, hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()

        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim),
                                                       nn.LeakyReLU()))
            
        self.shared_module = [self.critic_encoders]
        self.shared_modules = [self.key_extractors, self.selector_extractors, self.global_token_encoder, self.value_extractors]

    def shared_parameters(self):
        return chain(*[m.parameters() for m in self.shared_modules])
    
    def shared_parameter(self):
        return chain(*[m.parameters() for m in self.shared_module])

    def scale_shared_grads(self):
        if self.nagents > 1:
            for p in self.shared_parameter(): 
                if p.grad is not None: 
                    p.grad.data.mul_(1. / (self.nagents - 1))
                elif p.requires_grad:
                    warnings.warn(f"Parameter in shared_parameter() has no gradient... (Size: {p.shape})")

        if self.nagents > 0: 
            for p in self.shared_parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1. / self.nagents)
                elif p.requires_grad:
                    warnings.warn(f"Parameter in shared_parameter() has no gradient... (Size: {p.shape})")

                        
    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0,
                ego_states=None):
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        ego_in = states if ego_states is None else ego_states
        s_encodings = [self.state_encoders[a_i](ego_in[a_i]) for a_i in agents]
        
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        
        # calculate attention per head
        for head_i, (curr_head_keys, curr_head_values, curr_head_selectors) in enumerate(zip(
                all_head_keys, all_head_values, all_head_selectors)):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                
                # 🟢 [MSF-Token change] 1. Stack the SAEs of the other agents (excluding self)
                other_saes = [sa_encodings[j] for j in range(self.nagents) if j != a_i]
                
                if self.nagents > 1:
                    stacked_other_saes = torch.stack(other_saes)
                    # Compute mean and safe standard deviation (std)
                    mean_sae = torch.mean(stacked_other_saes, dim=0)
                    var_sae = torch.var(stacked_other_saes, dim=0, unbiased=False)
                    std_sae = torch.sqrt(var_sae + 1e-8)
                else:
                    mean_sae = torch.zeros_like(sa_encodings[0])
                    std_sae = torch.zeros_like(sa_encodings[0])
                
                # 🟢 [MSF-Token change] 2. Concat Mean and Std to produce 2*hidden_dim
                msf_token_input = torch.cat((mean_sae, std_sae), dim=1)
                
                # 🟢 [MSF-Token change] 3. Pass through the Global Token encoder (output dim: hidden_dim)
                global_token = self.global_token_encoder(msf_token_input)
                
                # 4. Extract the Global Token's Key and Value with the current head's linear layers
                global_key = self.key_extractors[head_i](global_token)
                global_value = self.value_extractors[head_i](global_token)

                # Copy the existing Key/Value lists of the other agents
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]

                # 5. Append the generated Global MSF virtual token to the end of the Key/Value lists
                keys.append(global_key)
                values.append(global_value)

                # Compute attention (virtual token included, so target length is N)
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

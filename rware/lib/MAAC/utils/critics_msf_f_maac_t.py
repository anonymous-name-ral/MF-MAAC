import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions + Global Mean Field Token.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1,
                 ego_dim=None):
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        # [Token change] Network that encodes the mean field into a virtual-token key/value
        
        self.global_token_encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(),
        )
        

        # Cosine Dissimilarity layer
        self.cosdis = nn.Linear(self.nagents * hidden_dim, hidden_dim)

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
            
            # [Dim changed] Mean field is now a token, so 4 * hidden_dim -> 3 * hidden_dim
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(3 * hidden_dim, hidden_dim * 2)) 
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim * 2, odim))
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

        # Update the shared parameter list (sae_mean_encoder -> global_token_encoder)
        self.shared_modules = [self.key_extractors, self.selector_extractors, self.cosdis,
                               self.value_extractors, self.critic_encoders ,self.global_token_encoder]

    def shared_parameters(self):
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        for p in self.shared_parameters():
            if p.grad is not None:
                p.grad.data.mul_(1. / self.nagents)

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
                
                # 🟢 [MF-Token] 1. Compute the mean field of the other agents (excluding self)
                other_saes = [sa_encodings[j] for j in range(self.nagents) if j != a_i]
                if self.nagents > 1:
                    stacked_other_saes = torch.stack(other_saes)
                    mean_sae = torch.mean(torch.stack(other_saes), dim=0)
                    var_sae = torch.var(stacked_other_saes, dim=0, unbiased=False)
                    std_sae = torch.sqrt(var_sae + 1e-8)
                else:
                    mean_sae = torch.zeros_like(sa_encodings[0])
                    std_sae = torch.zeros_like(sa_encodings[0])

                msf_token_input = torch.cat((mean_sae,std_sae),dim=1)

                # 🟢 [MF-Token] 2. Encode the Global Mean Token and extract Key/Value
                global_token = self.global_token_encoder(msf_token_input)
                global_key = self.key_extractors[head_i](global_token)
                global_value = self.value_extractors[head_i](global_token)

                # Get the existing Key/Value lists of the other agents
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]

                # 🟢 [MsF-Token] 3. Append the virtual token (number of targets: N)
                keys.append(global_key)
                values.append(global_value)

                # Attention computation
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # calculate cos dissimilarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_dis_list = []
        for i in range(self.nagents):
            cos_dis = []
            for j in range(self.nagents):
                val = cos(sa_encodings[i], sa_encodings[j]).unsqueeze(1) * -1
                val[val < 0] = 0 
                cos_dis.append(val)
            cos_dis_list.append(cos_dis)

        cos_dis_layer_in_all = []
        for i in range(self.nagents):
            cos_dis_layer_in = torch.cat(
                [cos_dis_list[i][j] * sa_encodings[j] for j in range(self.nagents)], dim=1)
            cos_dis_layer_in_all.append(cos_dis_layer_in)

        cos_dis_layer_out_all = []
        for i in range(self.nagents):
            cos_dis_layer_out_all.append(self.cosdis(cos_dis_layer_in_all[i]))

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            
            # [Input changed] agent_mv removed; concatenate the outputs of 3 modules
            critic_in = torch.cat((s_encodings[i], *other_all_values[i], cos_dis_layer_out_all[i]), dim=1)
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

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CylindricalMFMixer(nn.Module):
    """
    Implementation of Sec. 3.2 "Cylindrical Mixing Network" from the MF-QMIX paper.

    Key differences from QMixer (modules/mixers/qmix.py):
      - In QMixer, the hypernetwork directly generates weights (embed_dim * n_agents) for the n_agents
        individual Q-values -> parameters/compute scale with the number of agents (O(n)).
      - CylindricalMFMixer first summarizes the n_agents individual Q-values into a "moment vector" (fixed size K),
        and the hypernetwork only takes the "mean-field state" (fixed size), so regardless of agent count
        the number of parameters and the forward structure stay the same (O(1)).

    * The paper does not give an exact form for the moment function g_k, so we use the most standard reading,
      raw power moments g_k(Q) = Q^k (k=1..K).
    """

    def __init__(self, args):
        super(CylindricalMFMixer, self).__init__()
        self.args = args

        # Dimension of the mean-field state (mu_s) -- must be fixed regardless of agent count.
        # main_mfqmix.py computes it as "one obs_dim" and passes it in (get_state() averaged over the agent dim).
        self.mean_field_state_dim = args.mean_field_state_dim

        self.n_moments = getattr(args, "n_moments", 4)      # K: number of moment orders
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = args.hypernet_embed

        # 1. MLP mapping the moment vector (K dims) -> embedding (embed_dim dims)
        #    (paper: "moment embeddings are processed through a series of linear
        #     transformations and activation functions")
        self.moment_embed = nn.Sequential(
            nn.Linear(self.n_moments, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        
        '''
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.mean_field_state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * self.embed_dim),
        )
        '''
        
        self.hyper_w1 = nn.Linear(self.mean_field_state_dim, self.embed_dim * self.embed_dim)

        # 2. Hypernetwork: mean-field state (mu_s) -> mixing weights
        #    As in QMIX, abs() activation is used to preserve monotonicity.
        
        self.hyper_b1 = nn.Linear(self.mean_field_state_dim, self.embed_dim)

        
        '''
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.mean_field_state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim)
        )
        '''
        
        self.hyper_w_final = nn.Linear(self.mean_field_state_dim, self.embed_dim)

        # State-dependent bias (V(s)) -- same role as in QMIX
        self.V = nn.Sequential(
            nn.Linear(self.mean_field_state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def compute_moments(self, agent_qs):
        """
        agent_qs: (bs, n_agents) -- individual agent Q-values
        Returns: (bs, n_moments) -- fixed-size moment vector mu_Q_hat, independent of agent count
        """
        if self.n_moments >= 2:
            q_use = F.softplus(agent_qs)
        else:
            q_use = agent_qs
        moments = [(q_use ** k).mean(dim=1, keepdim=True) for k in range(1, self.n_moments + 1)]
        return th.cat(moments, dim=1)

    def forward(self, agent_qs, mean_field_state):
        """
        agent_qs         : (bs, n_agents) -- individual agent Q-values (same input shape as the original QMixer)
        mean_field_state  : (bs, mean_field_state_dim) -- mean-field state mu_s (fixed size, independent of agent count)
        """
        bs = agent_qs.size(0)

        # 1. Summarize individual Q-values into a fixed-size moment vector (n_agents information is dropped here)
        mu_q = self.compute_moments(agent_qs)                 # (bs, n_moments)
        mu_q_hat = self.moment_embed(mu_q)                    # (bs, embed_dim)

        # 2. Mean-field-state hypernetwork -> mixing weights (same abs() monotonicity constraint as QMIX)
        w1 = th.abs(self.hyper_w1(mean_field_state)).view(bs, self.embed_dim, self.embed_dim)
        b1 = self.hyper_b1(mean_field_state).view(bs, 1, self.embed_dim)
        hidden = F.elu(th.bmm(mu_q_hat.unsqueeze(1), w1) + b1)   # (bs, 1, embed_dim)

        w_final = th.abs(self.hyper_w_final(mean_field_state)).view(bs, self.embed_dim, 1)
        v = self.V(mean_field_state).view(bs, 1, 1)

        q_tot = th.bmm(hidden, w_final) + v                   # (bs, 1, 1)
        return q_tot.view(bs, -1, 1)
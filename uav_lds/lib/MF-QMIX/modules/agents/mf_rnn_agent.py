import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFieldRNNAgent(nn.Module):
    """
    Implementation of Sec. 3.1 "Mean-Field Agent Network" from the MF-QMIX paper.

    Differences from the original RNNAgent:
        the input is not just obs_i but
        [obs_i, last_action_i(one-hot), mean_field_obs, mean_field_action(one-hot)]
        a concatenation of four components.

    * mean_field_obs / mean_field_action are "averages over neighboring agents in the group";
      this implementation assumes a single homogeneous group with parameter sharing and
      simplifies "neighbors = the whole group including oneself" (the paper gives no exact definition).
      That is, every agent in the group receives the same mean_field_obs/mean_field_action values.
    """

    def __init__(self, input_shape, args):
        super(MeanFieldRNNAgent, self).__init__()
        self.args = args
        self.obs_dim = input_shape
        self.n_actions = args.n_actions

        # obs + last_action(one-hot) + mean_field_obs + mean_field_action(one-hot) 
        # ********Changed: add Agent ID (one-hot) dimension*******************
        combined_input_dim = self.obs_dim + self.n_actions + self.obs_dim + self.n_actions + args.n_agents
        
        self.fc1 = nn.Linear(combined_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    # ********Changed: add agent_id_onehot to input parameters************
    def forward(self, obs, last_action_onehot, mean_field_obs, mean_field_action, agent_id_onehot, hidden_state):
        """
        obs                : (..., obs_dim)
        last_action_onehot : (..., n_actions)  — one-hot of this agent's action at the previous timestep
        mean_field_obs      : (..., obs_dim)   — mean observation within the group (same value broadcast to all agents in the group)
        mean_field_action   : (..., n_actions) — mean previous-action distribution within the group (probability vector summing to 1)
        hidden_state         : (..., rnn_hidden_dim)

        All "..." dims (batch, n_agents, etc.) are handled flexibly, as in the QPLEX/QMIX RNNAgent.
        """
        orig_shape = obs.shape[:-1]

        obs = obs.reshape(-1, obs.shape[-1])
        last_action_onehot = last_action_onehot.reshape(-1, last_action_onehot.shape[-1])
        mean_field_obs = mean_field_obs.reshape(-1, mean_field_obs.shape[-1])
        mean_field_action = mean_field_action.reshape(-1, mean_field_action.shape[-1])
        agent_id_onehot = agent_id_onehot.reshape(-1, agent_id_onehot.shape[-1]) # *****Added
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        # *************Changed: concat agent_id_onehot as well
        inputs = torch.cat([obs, last_action_onehot, mean_field_obs, mean_field_action, agent_id_onehot], dim=-1)
        
        
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        q = q.view(*orig_shape, -1)
        h = h.view(*orig_shape, -1)
        return q, h
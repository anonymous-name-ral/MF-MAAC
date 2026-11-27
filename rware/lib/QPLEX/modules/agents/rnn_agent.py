import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
 
        # input_shape: pure obs_dim (agent ID one-hot is appended automatically in forward())
        # -> only fc1's input dim grows by n_agents
        # obs_agent_id=False (2026-10-22): obs-only input, as in the old-project QPLEX run1 (default True = unchanged)
        self.use_agent_id = getattr(args, 'obs_agent_id', True)
        self.fc1 = nn.Linear(input_shape + (self.n_agents if self.use_agent_id else 0), args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
 
    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
 
    def _append_agent_id(self, inputs):
        """
        inputs: (*lead_dims, n_agents, obs_dim)
            - in step(): lead_dims = ()          -> (n_agents, obs_dim)
            - in update(): lead_dims = (batch,)  -> (batch, n_agents, obs_dim)
        Appends the one-hot ID matching each agent position (agent axis) after obs and returns it.
        """
        lead_dims = inputs.shape[:-2]  # remaining dims excluding agent/obs axes (batch, etc.)
 
        eye = torch.eye(self.n_agents, device=inputs.device, dtype=inputs.dtype)
        eye = eye.view(*([1] * len(lead_dims)), self.n_agents, self.n_agents)
        agent_ids = eye.expand(*lead_dims, self.n_agents, self.n_agents)
 
        return torch.cat([inputs, agent_ids], dim=-1)  # (*lead_dims, n_agents, obs_dim+n_agents)
 
    def forward(self, inputs, hidden_state):
        orig_shape = inputs.shape[:-1]   # (batch, n_agents) or (n_agents,), etc.
 
        if self.use_agent_id:
            inputs = self._append_agent_id(inputs)             # attach agent ID one-hot
        inputs = inputs.reshape(-1, inputs.shape[-1])           # flatten to (N, obs_dim+n_agents)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  # (N, hidden_dim)
 
        x = F.relu(self.fc1(inputs))
        h = self.rnn(x, h_in)
        q = self.fc2(h)
 
        q = q.view(*orig_shape, -1)     # restore original shape: (batch, n_agents, n_actions)
        h = h.view(*orig_shape, -1)     # (batch, n_agents, hidden_dim)
        return q, h
 

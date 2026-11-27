import gym
import numpy as np
import rware  # noqa: F401  -- import to trigger environment registration


class RWARE_QPLEX_Wrapper(object):
    """
    Wrapper that adapts the RWARE (robotic-warehouse) environment to the interface
    expected by the QPLEX (main_qplex.py) training loop.

    Minimum interface required by the QPLEX side (main_qplex.py, qplex_agent.py):
        - self.nagent
        - self.observation_space : List[(obs_dim,)]  (per agent)
        - self.action_space_n    : List[n_actions]   (per agent)
        - reset() -> obs (List[np.ndarray])
        - step(actions) -> (next_obs, rewards, dones, info)
        - get_state() -> np.ndarray  (global state, all obs concatenated)
        - get_avail_actions() -> List[np.ndarray]  (action mask)
        - close()

    Notes specific to the warehouse.py structure:
        - reset() returns a 2-tuple (obs_tuple, info_dict) (new gym API)
        - step() returns a 5-tuple (obs, rewards, terminated, truncated, info)
        - action_space is plain Discrete(5) (msg_bits=0, so no one-hot conversion needed)
        - done(=terminated) becomes True only when max_steps (time limit) is reached
          (registered with max_inactivity_steps=None, so there is no "true termination" condition)
          -> i.e. this task is effectively a continuing task cut by a time limit, so
             the QPLEX buffer must treat it as "time-limit termination", not "true termination",
             so that TD target bootstrapping is not artificially cut off.
             (this is why main_qplex.py always records the terminated field as False)
    """

    def __init__(self, env_id, seed=None, render_mode="rgb_array"):
        self.env = gym.make(env_id, render_mode=render_mode)

        if seed is not None:
            self.env.seed(seed)

        self.nagent = self.env.n_agents

        # Per-agent observation dim (fast_obs flattened, so Box(flatdim,) shape)
        self.observation_space = [(space.shape[0],) for space in self.env.observation_space]

        # Per-agent number of actions (Discrete(5): NOOP/FORWARD/LEFT/RIGHT/TOGGLE_LOAD)
        self.action_space_n = [space.n for space in self.env.action_space]

        self.current_obs = None

    def reset(self):
        result = self.env.reset()
        # New gym API: returns (obs_tuple, info)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            obs, _info = result
        else:
            obs = result

        obs = [np.array(o, dtype=np.float32) for o in obs]
        self.current_obs = obs
        return obs

    def step(self, actions):
        """
        actions: integer index array/list of shape (n_agents,) (0~4)
                 -> RWARE needs no one-hot conversion, so pass through as-is
        """
        discrete_actions = [int(a) for a in actions]

        step_result = self.env.step(discrete_actions)

        if len(step_result) == 5:
            next_obs, rewards, terminated, truncated, info = step_result
        else:
            # Safeguard in case it runs with the old gym API (4-tuple)
            next_obs, rewards, terminated, info = step_result
            truncated = [False] * self.nagent

        next_obs = [np.array(o, dtype=np.float32) for o in next_obs]
        self.current_obs = next_obs

        # Pass truncated info in info (used in main_qplex.py for reference only;
        # the actual terminated decision is handled in main_qplex.py as described in the docstring above)
        info = dict(info) if isinstance(info, dict) else {}
        info["truncated"] = truncated

        # The QPLEX training loop takes done as a "list" and checks all(dones), so return it as-is
        return next_obs, rewards, terminated, info

    def get_state(self):
        """Global state = all agents' observations concatenated"""
        return np.concatenate(self.current_obs, axis=0)

    def get_avail_actions(self):
        """RWARE has no action masking -> always all available (1)"""
        avail_actions = []
        for n_action in self.action_space_n:
            avail_actions.append(np.ones(n_action, dtype=np.float32))
        return avail_actions

    def close(self):
        self.env.close()
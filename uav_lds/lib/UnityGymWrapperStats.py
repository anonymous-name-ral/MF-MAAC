
from typing import Optional
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

from mlagents_envs.base_env import ActionTuple

"""
# 
# Unity Env Gym Wrapper for multi-agents (+ StatsSideChannel delivery counters, 2026-10-19) (version 5)
# for mlagents 0.27.0 (Unity API version 2.1.0 exp1) 
#
"""

class GymEnv(object):

    # GymEnv Object <- (environment)
    def __init__(self, 
                 name : str, 
                 mapsize : Optional[int] = 13, 
                 numbuilding : Optional[int] = 3, 
                 max_smallbox : Optional[int] = 10,
                 max_bigbox : Optional[int] = 10,
                 width : Optional[int] = 480, 
                 height : Optional[int] = 270, 
                 timescale : Optional[float] = 20,
                 quality_level : Optional[int] = 5,
                 target_frame_rate : Optional[int] = None,
                 capture_frame_rate : Optional[int] = None,
                 worker_id : int = 0,
                 no_graphics : bool = False,
                 seed : int = 0
                 ):
        """
        <Logistics Parameters>
        name : path to Unity Environment
        mapsize : map size (x by x)
        numbuilding : number of buildings
        max_smallbox : max number of small parcels to check time
        max_bigbox : max number of big parcels to check time

        <Unity Environment Engine Parameters>
        width : Defines the width of the display. (Must be set alongside height)
        height : Defines the height of the display. (Must be set alongside width)
        timescale : Defines the multiplier for the deltatime in the simulation. 
                    If set to a higher value, time will pass faster in the simulation 
                    but the physics may perform unpredictably.
        quality_level : Defines the quality level of the simulation.
        target_frame_rate : Instructs simulation to try to render at a specified frame rate.
        capture_frame_rate : Instructs the simulation to consider time between updates to 
                             always be constant, regardless of the actual frame rate.
        """

        # [2026-10-19] episode counters exported by Build_Linux_eval (StatsSideChannel); 0 on older builds
        self._stats = StatsSideChannel()
        self.counters = {'small_delivered': 0, 'big_delivered': 0, 'collisions': 0}

        # engine channels
        engine_channel = EngineConfigurationChannel()

        # environment channels
        env_channel = EnvironmentParametersChannel()


        self.env = UnityEnvironment(
            file_name=name, 
            worker_id=worker_id + 1003, 
            no_graphics=no_graphics, 
            seed=seed,              # [multi-seed 2026-11-11] Unity RNG seed (default 0 = ML-Agents default, as before)
            side_channels=[engine_channel, env_channel, self._stats]
        )
        
        engine_channel.set_configuration_parameters(
            width=width, 
            height=height, 
            time_scale=timescale, 
            quality_level=quality_level,
            target_frame_rate=target_frame_rate,
            capture_frame_rate=capture_frame_rate)

        env_channel.set_float_parameter("mapsize", mapsize)
        env_channel.set_float_parameter("building_num", numbuilding)
        env_channel.set_float_parameter("slimit", max_smallbox)
        env_channel.set_float_parameter("blimit", max_bigbox)
        
        self.env.reset()

        # behavior names
        self.behavior_names = list(self.env.behavior_specs.keys())
        self.nbehavior = len(self.behavior_names)

        print('-' * 20)
        print("behavior names : ")
        print(self.behavior_names)
        print("number of behaviors :", self.nbehavior)

        # behavior specs
        self.specs = []
        for behavior_name in self.behavior_names:
            print("behavior spec of " + behavior_name + " :")
            print(self.env.behavior_specs[behavior_name])
            self.specs.append(self.env.behavior_specs[behavior_name])

        self.decision_steps = []
        self.terminal_steps = []

        for _i in range(self.nbehavior):
            d_s, t_s = self.env.get_steps(self.behavior_names[_i])
            self.decision_steps.append(d_s)
            self.terminal_steps.append(t_s)

        self.n_each_agent = []
        
        for _i in range(self.nbehavior):
            self.n_each_agent.append(len(self.decision_steps[_i]))

        print('-' * 20)
        print("number of agents for each behavior :")
        print(self.n_each_agent)

        self.nagent = sum(self.n_each_agent)
        self.agents = [None] * self.nagent

        print('-' * 20)
        print("nagent (number of agents) :", self.nagent)

        self.observation_space = []
        self.action_space = []
        self.action_space_n = []

        for _i in range(self.nbehavior):
            for _j in range(self.n_each_agent[_i]):
                # self.observation_space.append(self.specs[_i].observation_shapes[0])
                self.observation_space.append(self.reshape_obs(self.decision_steps[_i])[0].shape)
                self.action_space.append(len(self.env.behavior_specs[self.behavior_names[_i]].action_spec.discrete_branches))
                self.action_space_n.append(len(self.env.behavior_specs[self.behavior_names[_i]].action_spec.discrete_branches))

        self.discrete_branches = []
        for _i in range(self.nbehavior):
            branches = self.env.behavior_specs[self.behavior_names[_i]].action_spec.discrete_branches
            for _j in range(self.n_each_agent[_i]):
                self.discrete_branches.append(branches)

        print('-' * 20)
        print("observation space")
        print(self.observation_space)

        print('-' * 20)
        print("action space")
        print(self.action_space)

        print('-' * 20)
        print("GymWrapper Initialized Done!")
        print('-' * 20)


    def reshape_obs(self, d_s):
        """
        Converting each multi-dimensional obs arrays to one-dimensional obs arrays
        Use this to conc obs and ray-cast obs to one-dimensional array.
        """
        return np.column_stack(tuple(d_s.obs))
    

    # obs <- reset(None)
    def reset(self):
        """
        Reset Environment. Returns initial obs.
        """
        self.env.reset()
        self._stats.get_and_reset_stats()
        self.counters = {k: 0 for k in self.counters}
        self.decision_steps = []
        self.terminal_steps = []

        for _i in range(self.nbehavior):
            d_s, t_s = self.env.get_steps(self.behavior_names[_i])
            self.decision_steps.append(d_s)
            self.terminal_steps.append(t_s)

        obs = []

        for _i in range(self.nbehavior):
            for o in self.reshape_obs(self.decision_steps[_i]):
                obs.append(o)

        self.current_obs=obs

        return obs


    # obs, reward, done, info <- step(actions)
    def step(self, actions):
        """
        make a single step of environment.
        requires actions of all agents, returns obs, rewards, dones, infos like Gym.
        """
        
        lastidx = 0
        for _i in range(self.nbehavior):
            action_tuple = ActionTuple()
            action_tuple.add_discrete(actions[lastidx:lastidx + self.n_each_agent[_i], :])
            self.env.set_actions(behavior_name=self.behavior_names[_i], action=action_tuple)
            lastidx = self.n_each_agent[_i]

        self.env.step()
        self.decision_steps = []
        self.terminal_steps = []

        for _i in range(self.nbehavior):
            d_s, t_s = self.env.get_steps(self.behavior_names[_i])
            self.decision_steps.append(d_s)
            self.terminal_steps.append(t_s)

        obs = []
        reward = []
        done = []
        info = {}

        for _i in range(self.nbehavior):
            _j = 0
            for o in self.reshape_obs(self.decision_steps[_i]):
                obs.append(o)

                
                #raw_reward = self.decision_steps[_i].reward[_j]
                #if raw_reward > 1.0:
                #    print(f"\n[Wrapper Check] Behavior {_i}, Agent {_j} Raw Reward: {raw_reward}")
                    
                    
                reward.append(self.decision_steps[_i].reward[_j])  
                done.append(False)
                _j += 1
        
        self.current_obs = obs
        for k, v in self._stats.get_and_reset_stats().items():
            name = k.split('/')[-1]
            if name in self.counters and v:
                self.counters[name] = int(round(v[-1][0]))      # cumulative episode count (MostRecent)
        info = dict(self.counters)

        return obs, reward, done, info
    
    def get_state(self):
        """
        returns the global state bt concatenating all agents' observations
        """        
        state = np.concatenate(self.current_obs, axis=0)
        return state
    
    def get_avail_actions(self):
        """
        LogisitcEnv is avilable on all actions
        """
        avail_actions = []
        for n_action in self.action_space_n:
            avail_actions.append(np.ones(n_action, dtype=np.float32))
        return avail_actions
        

    # None <- close(None)
    def close(self):
        """
        Close Environment
        """
        self.env.close()


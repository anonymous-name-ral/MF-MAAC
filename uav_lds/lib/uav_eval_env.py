"""
UAV-LDS evaluation wrapper. Same interface as UnityGymWrapperStats.GymEnv (list of 84-dim observations,
actions (n_agents, 7) one-hot over the 7 discrete branches), plus:
  - reset(seed): sets the Unity environment parameter "seed" (Random.InitState before the episode is built:
    map, hubs, buildings, UAV starts, parcel destinations)
  - step() info: episode counters exported by Unity through the StatsSideChannel:
      small_delivered, big_delivered, collisions_total (UAV-UAV pairs + building hits), collisions_uav,
      collisions_building, per_uav {UAV<k>: {'uav': n, 'building': n}}; 'collisions' = building hits only
  - spawn_xz: half-width of the UAV start area, sent only when given (default of the build: 2)
Engine settings default to the training ones (time scale 20, quality 1, no graphics).
Python agent order is not by name: index 0..4 = UAV1, UAV2, UAV4, UAV5, UAV3.
"""
import re

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel

KEYS = {'UAVLDS/small_delivered': 'small_delivered', 'UAVLDS/big_delivered': 'big_delivered',
        'UAVLDS/collisions': 'collisions'}
V2_KEYS = {'UAVLDS/collisions_total': 'collisions_total', 'UAVLDS/collisions_uav': 'collisions_uav',
           'UAVLDS/collisions_building': 'collisions_building'}
PER_UAV = re.compile(r'^UAVLDS/(UAV\d+)/collide_(uav|building)$')


class UAVEvalEnv(object):
    def __init__(self, build, worker_id=0, mapsize=13, numbuilding=3, max_smallbox=100, max_bigbox=100,
                 timescale=20, quality_level=1, width=480, height=270, no_graphics=True, unity_seed=0,
                 target_frame_rate=None, capture_frame_rate=None, spawn_xz=None):
        self.engine = EngineConfigurationChannel()
        self.params = EnvironmentParametersChannel()
        self.stats = StatsSideChannel()
        self.env = UnityEnvironment(file_name=build, worker_id=worker_id, no_graphics=no_graphics,
                                    seed=unity_seed, side_channels=[self.engine, self.params, self.stats])
        self.engine.set_configuration_parameters(width=width, height=height, time_scale=timescale,
                                                 quality_level=quality_level, target_frame_rate=target_frame_rate,
                                                 capture_frame_rate=capture_frame_rate)
        self.params.set_float_parameter("mapsize", mapsize)
        self.params.set_float_parameter("building_num", numbuilding)
        self.params.set_float_parameter("slimit", max_smallbox)
        self.params.set_float_parameter("blimit", max_bigbox)
        self.spawn_xz = spawn_xz
        if spawn_xz is not None:                                   # only when given (5 UAVs: never)
            self.params.set_float_parameter("spawn_xz", float(spawn_xz))
        self.env.reset()
        self.behavior = list(self.env.behavior_specs.keys())[0]
        spec = self.env.behavior_specs[self.behavior]
        d_s, _ = self.env.get_steps(self.behavior)
        self.nagent = len(d_s)
        self.n_branches = len(spec.action_spec.discrete_branches)
        self.obs_dim = self._obs(d_s)[0].shape[0]
        self.counters = {}

    @staticmethod
    def _obs(d_s):
        return [o for o in np.column_stack(tuple(d_s.obs))]

    def _read_stats(self):
        for k, v in self.stats.get_and_reset_stats().items():
            if not v:
                continue
            val = int(round(v[-1][0]))                             # MostRecent value of this step
            if k in KEYS:
                self.counters[KEYS[k]] = val
            elif k in V2_KEYS:
                self.counters[V2_KEYS[k]] = val
            else:
                m = PER_UAV.match(k)
                if m:
                    self.counters.setdefault('per_uav', {}).setdefault(m.group(1), {})[m.group(2)] = val

    def reset(self, seed=None):
        if seed is not None:
            self.params.set_float_parameter("seed", float(seed))
        self.env.reset()
        self.stats.get_and_reset_stats()
        self.counters = {k: 0 for k in KEYS.values()}          # v2 keys appear with the first step (v2 builds)
        d_s, _ = self.env.get_steps(self.behavior)
        return self._obs(d_s)

    def step(self, actions):
        self.env.set_actions(self.behavior, ActionTuple(discrete=np.asarray(actions, dtype=np.int32)))
        self.env.step()
        d_s, _ = self.env.get_steps(self.behavior)
        self._read_stats()
        info = dict(self.counters)
        if 'per_uav' in info:
            info['per_uav'] = {u: dict(c) for u, c in info['per_uav'].items()}
        return self._obs(d_s), list(d_s.reward), [False] * self.nagent, info

    def close(self):
        self.env.close()

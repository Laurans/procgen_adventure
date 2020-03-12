import gym
import numpy as np
from gym import ActionWrapper, ObservationWrapper
from gym.spaces import Box
from procgen import ProcgenEnv


def make(**env_config):
    env = ProcgenEnv(**env_config)
    env = EpisodeRewardWrapper(env)
    env = RemoveDictObs(env, key="rgb")
    env = ReshapeAction(env)
    return env


class RemoveDictObs(ObservationWrapper):
    def __init__(self, env, key):
        self.key = key
        super().__init__(env=env)
        self.observation_space = env.observation_space.spaces[self.key]

    def observation(self, obs):
        return obs[self.key]


class PermuteShapeObservation(ObservationWrapper):
    def __init__(self, env):
        super(PermuteShapeObservation, self).__init__(env)

        assert len(env.observation_space.shape) == 3
        obs_shape = self.observation_space.shape[:2]
        channel = self.observation_space.shape[2]
        self.observation_space = Box(
            low=0, high=255, shape=(channel, *obs_shape), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


class ReshapeAction(ActionWrapper):
    def __init__(self, env):
        super(ReshapeAction, self).__init__(env)
        self.mapping_action = {
            "NOOP": 4,
            "RIGHT": 7,
            "LEFT": 1,
            "JUMP": 5,
            "RIGHT-JUMP": 8,
            "LEFT-JUMP": 2,
            "DOWN": 3,
        }
        self.action_space = gym.spaces.Discrete(len(self.mapping_action.keys()))

    def action(self, action):
        actions = [
            self.mapping_action[list(self.mapping_action)[act]] for act in action
        ]
        return np.array(actions)


class EpisodeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        env.metadata = {"render.modes": []}
        env.reward_range = (-float("inf"), float("inf"))
        nenvs = env.num_envs
        self.num_envs = nenvs
        super(EpisodeRewardWrapper, self).__init__(env)

        self.aux_rewards = None
        self.num_aux_rews = None

        def reset(**kwargs):
            self.rewards = np.zeros(nenvs)
            self.lengths = np.zeros(nenvs)
            self.aux_rewards = None
            self.long_aux_rewards = None

            return self.env.reset(**kwargs)

        def step(action):
            obs, rew, done, infos = self.env.step(action)

            if self.aux_rewards is None:
                info = infos[0]
                if "aux_rew" in info:
                    self.num_aux_rews = len(infos[0]["aux_rew"])
                else:
                    self.num_aux_rews = 0

                self.aux_rewards = np.zeros(
                    (nenvs, self.num_aux_rews), dtype=np.float32
                )
                self.long_aux_rewards = np.zeros(
                    (nenvs, self.num_aux_rews), dtype=np.float32
                )

            self.rewards += rew
            self.lengths += 1

            use_aux = self.num_aux_rews > 0

            if use_aux:
                for i, info in enumerate(infos):
                    self.aux_rewards[i, :] += info["aux_rew"]
                    self.long_aux_rewards[i, :] += info["aux_rew"]

            for i, d in enumerate(done):
                if d:
                    epinfo = {
                        "r": round(self.rewards[i], 6),
                        "l": self.lengths[i],
                        "t": 0,
                    }
                    aux_dict = {}

                    for nr in range(self.num_aux_rews):
                        aux_dict["aux_" + str(nr)] = self.aux_rewards[i, nr]

                    if "ale.lives" in infos[i]:
                        game_over_rew = np.nan

                        is_game_over = infos[i]["ale.lives"] == 0

                        if is_game_over:
                            game_over_rew = self.long_aux_rewards[i, 0]
                            self.long_aux_rewards[i, :] = 0

                        aux_dict["game_over_rew"] = game_over_rew

                    epinfo["aux_dict"] = aux_dict

                    infos[i]["episode"] = epinfo

                    self.rewards[i] = 0
                    self.lengths[i] = 0
                    self.aux_rewards[i, :] = 0

            return obs, rew, done, infos

        self.reset = reset
        self.step = step

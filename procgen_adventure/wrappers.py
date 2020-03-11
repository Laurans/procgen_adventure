import gym
import numpy as np
from gym import ObservationWrapper, RewardWrapper, Wrapper, ActionWrapper
from gym.spaces import Box


def _make(**env_config):
    env = gym.make(**env_config)
    env = PermuteShapeObservation(env)
    env = TimeLimit(env, 100)
    # env = ReshapeReward(env)
    env = ReshapeAction(env)
    return env


def make(*args, **env_config):
    """Use as factory function for making instances of gym environment with
    rlpyt's ``GymEnvWrapper``, using ``gym.make(*args, **kwargs)``.  If
    ``info_example`` is not ``None``, will include the ``EnvInfoWrapper``.
    """
    env = _make(**env_config)
    env = RestartEnv(env, env_config)
    return env


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


class RestartEnv(Wrapper):
    def __init__(self, env, env_config):
        super(RestartEnv, self).__init__(env)
        self.env_config = env_config

    def reset(self, **kwargs):
        self.env = _make(**self.env_config)
        return super().reset(**kwargs)


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ReshapeReward(RewardWrapper):
    def reward(self, reward):
        if reward >= 10:
            return reward
        else:
            return reward - 0.1


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
        return self.mapping_action[list(self.mapping_action)[action]]

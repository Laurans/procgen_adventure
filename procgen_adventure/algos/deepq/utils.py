import numpy as np
from procgen_adventure.utils.collections import namedarraytuple

SamplesToBuffer = namedarraytuple(
    "SamplesToBuffer", ["observation", "action", "reward", "done"]
)


def get_example_outputs(env):
    env.reset()
    a = env.action_space.sample()
    if env.num_envs > 1:
        o, r, d, env_info = env.step([a] * env.num_envs)
        o = o[0]
        r = r[:1]
        d = d[:1]
        a = np.array([a])
    else:
        breakpoint()
    examples = dict(observation=o, reward=r, done=d, action=a)
    return samples_to_buffer(examples)


def samples_to_buffer(samples: dict):
    return SamplesToBuffer(
        observation=samples["observation"],
        action=samples["action"],
        reward=samples["reward"],
        done=samples["done"],
    )

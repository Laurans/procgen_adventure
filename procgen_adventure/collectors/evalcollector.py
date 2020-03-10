import time

import numpy as np
from rlpyt.samplers.collectors import BaseEvalCollector
from rlpyt.utils.buffer import buffer_from_example, numpify_buffer, torchify_buffer
from rlpyt.utils.quick_args import save__init__args

# For sampling, serial sampler can use Cpu collectors.


class EvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
        self, env, agent,
    ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        observations = list()
        observations.append(self.env.reset())
        observation = buffer_from_example(observations[0], 1)
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.env.action_space.null_value(), 1)
        reward = np.zeros(1, dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        done = False
        while not done:
            self.env.render("rgb_array")
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            print(action)
            o, r, done, env_info = self.env.step(action[0])
            observation[b] = o
            reward[b] = r
            time.sleep(0.01)

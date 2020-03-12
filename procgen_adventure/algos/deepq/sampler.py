from collections import defaultdict

import numpy as np
import torch


from procgen_adventure.utils.torch_utils import input_preprocessing, to_np
from typing import Callable


class Sampler:
    def __init__(
        self,
        env,
        model,
        replay,
        num_steps,
        device,
        num_envs,
        exploration_steps,
        random_action_prob: Callable,
    ):
        self.env = env
        self.action_n = env.action_space.n
        self.model = model
        self.num_steps = num_steps
        self.device = device

        self.obs = np.zeros(
            (num_envs,) + env.observation_space.shape,
            dtype=env.observation_space.dtype.name,
        )

        self.obs[:] = env.reset()
        self.dones = np.array([False] * num_envs)
        self._total_steps = 0
        self.exploration_steps = exploration_steps
        self.random_action_prob = random_action_prob

    def interact(self, num_steps=1):
        storage = defaultdict(list)
        epinfos = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_steps):
                if (
                    self._total_steps < self.exploration_steps
                    or np.random.rand() < self.random_action_prob()
                ):
                    actions = np.random.randint(0, self.action_n)
                else:
                    obs = input_preprocessing(self.obs, device=self.device)
                    q_values = to_np(self.model.step(obs=obs)["value"])
                    breakpoint()
                    actions = np.argmax(q_values, axis=0)

                storage["states"] += [to_np(self.obs.clone())]
                storage["actions"] += [actions]

                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                storage["rewards"] += [rewards]
                storage["dones"] += [self.dones]
                for info in infos:
                    if "episode" in info:
                        epinfos.append(info["episode"])

        # batch of steps to batch of rollouts
        for key in storage:
            storage[key] = np.asarray(storage[key])

    def run(self):
        pass

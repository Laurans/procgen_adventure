from collections import defaultdict
from typing import Callable

import numpy as np
import torch

from procgen_adventure.utils.torch_utils import input_preprocessing, to_np
from procgen_adventure.replays.non_sequence.n_step import SamplesFromReplay
from .utils import samples_to_buffer


class Sampler:
    def __init__(
        self,
        env,
        model,
        replay,
        device,
        exploration_steps,
        random_action_prob: Callable,
        batch_size: int,
        sgd_update_freq: int,
    ):
        self.env = env
        self.action_n = env.action_space.n
        self.model = model
        self.replay_buffer = replay
        self.device = device
        self.batch_size = batch_size
        self.sgd_update_freq = sgd_update_freq

        self.obs = np.zeros(
            (env.num_envs,) + env.observation_space.shape,
            dtype=env.observation_space.dtype.name,
        )

        self.exploration_steps = exploration_steps // env.num_envs
        self.random_action_prob = random_action_prob

    def interact(self, random_act_only=False):
        storage = defaultdict(list)
        epinfos = []
        self.model.eval()
        with torch.no_grad():
            if random_act_only or np.random.rand() < self.random_action_prob():
                actions = np.random.randint(0, self.action_n, size=self.env.num_envs)
            else:
                obs = input_preprocessing(self.obs, device=self.device)
                q_values = to_np(self.model.step(obs=obs)["value"])
                breakpoint()
                actions = np.argmax(q_values, axis=0)

            storage["observation"] += [self.obs.copy()]
            storage["action"] += [actions]

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            storage["reward"] += [rewards]
            storage["done"] += [self.dones]
            for info in infos:
                if "episode" in info:
                    epinfos.append(info["episode"])

        # batch of steps to batch of rollouts
        for key in storage:
            storage[key] = np.asarray(storage[key])
            if storage[key].ndim == 2:
                storage[key] = np.expand_dims(storage[key], -1)

        return storage

    def run_exploration_steps(self):
        self.init_run()
        for _ in range(self.exploration_steps):
            samples = self.interact(random_act_only=True)
            self.replay_buffer.append_samples(samples_to_buffer(samples))

    def init_run(self):
        self.obs[:] = self.env.reset()
        self.dones = np.array([False] * self.env.num_envs)
        self._total_steps = 0

    def run(self):
        for _ in range(self.sgd_update_freq):
            samples = self.interact()
            self.replay_buffer.append_samples(samples_to_buffer(samples))

        samples_from_replay: SamplesFromReplay = self.replay_buffer.sample_batch(
            self.batch_size
        )
        batch = dict(
            states=input_preprocessing(samples_from_replay.agent_inputs.observation),
            actions=samples_from_replay.action,
            rewards=samples_from_replay.return_,
            next_states=input_preprocessing(
                samples_from_replay.target_inputs.observation
            ),
            dones=samples_from_replay.done,
        )
        return batch

import os
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from gym import Wrapper
from procgen.interactive import ProcgenEnv, ProcgenInteractive, Scalarize


def visualize_image(img, non_blocking=False):
    plt.imshow(img)
    if non_blocking:
        plt.draw()
        plt.pause(0.01)
    else:
        plt.show()


class Episode:
    def __init__(self, observation):
        self.level_seed = None

        observation = preprocess_img(observation)
        self.data = {"observations": [observation], "actions": [], "rewards": []}

    def set_level_seed(self, level_seed):
        self.level_seed = level_seed

    def add(self, act, rew, next_obs, done):
        if not done:
            next_obs = preprocess_img(next_obs)
            self.data["observations"] += [next_obs]
        self.data["actions"] += [np.int8(act)]
        self.data["rewards"] += [np.int8(rew)]

    def export(self):
        return {k: np.array(t) for k, t in self.data.items()}

    def show(self):
        for img in self.data["observations"]:
            visualize_image(postprocess_img(img), non_blocking=True)


def postprocess_img(str_encode):
    nparr = np.frombuffer(str_encode, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_decode


def preprocess_img(img):
    str_encode = cv2.imencode(".png", img)[1].tostring()
    return str_encode


class Monitor(Wrapper):
    def __init__(self, env, env_name):
        super(Monitor, self).__init__(env)
        self.hdf5_path = "data.hdf5"
        self.hdf5_file = h5py.File(self.hdf5_path, mode="a")
        self.hdf5_file.attrs["env_name"] = env_name
        self.iterator = 0

    def step(self, action):
        observation, reward, done, info = super(Monitor, self).step(action)
        self._after_step(action, observation, reward, done, info)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super(Monitor, self).reset(**kwargs)
        self._after_reset(observation)
        return observation

    def close(self):
        self._before_close()
        super(Monitor, self).close()

    def _after_step(self, action, observation, reward, done, info):

        self.episode.add(action, reward, observation["rgb"], done)

        if done:
            if info["level_complete"]:
                self.episode.set_level_seed(info["level_seed"])
                self.save_episode()

    def _after_reset(self, observation):
        self.episode = Episode(observation["rgb"])

    def _before_close(self):
        self.hdf5_file.close()

    def save_episode(self):
        group = self.hdf5_file.create_group(f"level_{self.iterator}")

        for key, value in self.episode.export().items():
            group.create_dataset(
                key,
                shape=value.shape,
                dtype=value.dtype,
                data=value,
                chunks=True,
                compression="gzip",
            )
        group.attrs["level_seed"] = self.episode.level_seed

        self.hdf5_file.flush()
        self.iterator += 1


class ProcgenInteractiveRecorder(ProcgenInteractive):
    def __init__(self, vision, **kwargs):
        self._vision = vision
        venv = ProcgenEnv(num_envs=1, **kwargs)
        self.combos = list(venv.unwrapped.combos)
        self.last_keys = []
        env = Scalarize(venv)
        env = Monitor(env, kwargs["env_name"])
        super(ProcgenInteractive, self).__init__(
            env=env, sync=False, tps=15, display_info=True
        )

    def _update(self, dt):
        original = sys.stdout
        sys.stdout = open(os.devnull, "w")
        super(ProcgenInteractiveRecorder, self)._update(dt)
        sys.stdout = original


venv = ProcgenInteractiveRecorder(vision="human", env_name="coinrun")
venv.run()

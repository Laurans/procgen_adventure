import os
import random

import torch.distributed as dist

from procgen_adventure.config.ppo import PPOExpConfig
from procgen_adventure.env.wrappers import make
from procgen_adventure.ppo.entrypoint import PPO
from procgen_adventure.utils.context import logger_context


def multi_setup(rank, world_size, Config):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    env = make(**Config.ENV_CONFIG)

    with logger_context(run_ID=0, name="coinrun", rank=rank) as logger:
        runner = PPO(rank, env, Config, logger)
        runner.learn()


def run():
    rand_port = random.SystemRandom().randint(1000, 2000)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(rand_port)

    Config = PPOExpConfig()

    if Config.WORLD_SIZE > 1:
        pass
    else:
        multi_setup(0, 1, Config)


if __name__ == "__main__":
    run()

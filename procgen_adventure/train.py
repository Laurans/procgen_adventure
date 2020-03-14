import os
import random

import torch.distributed as dist
from procgen_adventure.utils.cmd_util import common_arg_parser

from procgen_adventure.utils.wrappers import make
from procgen_adventure.utils.context import logger_context
from importlib import import_module
from procgen_adventure.algos.deepq.entrypoint import DeepQ


def multi_setup(alg, rank, world_size, Config):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    env = make(**Config.ENV_CONFIG)

    with logger_context(run_ID=0, name="coinrun", rank=rank) as logger:
        runner = get_runner_cls(alg)(rank, env, Config, logger)
        runner.learn()


def train(alg):
    rand_port = random.SystemRandom().randint(1000, 2000)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(rand_port)

    Config = get_config(alg)()

    if Config.WORLD_SIZE > 1:
        pass
    else:
        multi_setup(alg, 0, 1, Config)


def get_runner_cls(alg):
    return get_alg_module(alg).get_alg_class()


def get_config(alg):
    return get_alg_module(alg).get_alg_config()


def get_alg_module(alg):

    alg_module = import_module(".".join(["procgen_adventure", "algos", alg]))

    return alg_module


if __name__ == "__main__":
    arg_parser = common_arg_parser()
    args = arg_parser.parse_args()
    if args.alg is not None:
        train(args.alg)

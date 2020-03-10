import procgen  # noqa
from procgen_adventure.agents.procgen_dqn_agent import ProcgenDqnAgent
from procgen_adventure.runners.minibatch_rl import MinibatchRlEval
from procgen_adventure.utils.context import logger_context
from procgen_adventure.wrappers import make as gym_make
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler


def build_and_train(game="procgen:procgen-coinrun-v0"):
    run_ID = 0
    cuda_idx = 0
    n_parallel = 4
    sample_mode = "serial"

    config = dict(
        env=dict(
            id=game,
            num_levels=1,
            start_level=32,
            distribution_mode="easy",
            paint_vel_info=True,
        ),
        sampler=dict(batch_T=1, batch_B=16),
        algo=dict(
            min_steps_learn=int(2e4),
            replay_size=int(1e5),
            target_update_interval=int(8e3),
            n_step_return=3,
            learning_rate=0.0000625,
            eps_steps=int(25e4),
            optim_kwargs=dict(eps=0.00015, weight_decay=0.0),
            double_dqn=True,
            prioritized_replay=True,
            replay_ratio=4,
        ),
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel))),
    )

    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
    elif sample_mode == "cpu":
        Sampler = CpuSampler

    sampler = Sampler(
        EnvCls=gym_make,
        env_kwargs=config["env"],
        eval_env_kwargs=dict(id=game),
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(2000 * 10),
        eval_max_trajectories=None,
        **config["sampler"]
    )
    algo = DQN(**config["algo"])  # Run with defaults.
    agent = ProcgenDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=config["affinity"],
    )
    name = "dqn_" + "coinrun"

    with logger_context(run_ID, name, snapshot_mode="last"):
        runner.train()


build_and_train()

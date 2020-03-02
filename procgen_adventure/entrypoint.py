import procgen  # noqa
from procgen_adventure.agents.procgen_dqn_agent import ProcgenDqnAgent
from procgen_adventure.runners.minibatch_rl import MinibatchRlEval
from procgen_adventure.wrappers import make as gym_make
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.samplers.serial.sampler import SerialSampler


def build_and_train(game="procgen:procgen-coinrun-v0", run_ID=0, cuda_idx=0):
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(
            id=game, num_levels=1, start_level=32, distribution_mode="easy"
        ),
        eval_env_kwargs=dict(id=game),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension)
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e3)  # Run with defaults.
    agent = ProcgenDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    # config = dict(game=game)
    # name = "dqn_" + game
    # log_dir = "example_1"
    # with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
    runner.train()


build_and_train()

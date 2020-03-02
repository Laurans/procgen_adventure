"""
Runs one instance of the Atari environment and optimizes using DQN algorithm.
Can use a GPU for the agent (applies to both sample and train). No parallelism
employed, so everything happens in one python process; can be easier to debug.

The kwarg snapshot_mode="last" to logger context will save the latest model at
every log point (see inside the logger for other options).

In viskit, whatever (nested) key-value pairs appear in config will become plottable
keys for showing several experiments.  If you need to add more after an experiment,
use rlpyt.utils.logging.context.add_exp_param().

"""

from procgen_adventure.runners.minibatch_rl import MinibatchRlEval
from procgen_adventure.utils.context import logger_context
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.samplers.serial.sampler import SerialSampler


def build_and_train(game="pong", run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(2700 * 10),
        eval_max_trajectories=None,
    )
    algo = DQN(
        min_steps_learn=int(2e4),
        replay_size=int(1e5),
        target_update_interval=int(8e3),
        n_step_return=3,
        learning_rate=0.0000625,
        eps_steps=int(25e4),
        optim_kwargs=dict(eps=0.00015, weight_decay=0.0),
        double_dqn=True,
    )
    agent = AtariDqnAgent(model_kwargs=dict(dueling=True))
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    name = "dqn_" + game
    with logger_context(run_ID, name, snapshot_mode="last"):
        runner.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--game", help="Atari game", default="pong")
    parser.add_argument(
        "--run_ID", help="run identifier (logging)", type=int, default=0
    )
    parser.add_argument("--cuda_idx", help="gpu to use ", type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        game=args.game, run_ID=args.run_ID, cuda_idx=args.cuda_idx,
    )

import torch
from procgen_adventure.agents.procgen_dqn_agent import ProcgenDqnAgent
from procgen_adventure.collectors.evalcollector import EvalCollector
from procgen_adventure.wrappers import make


def load_params(file_name):
    params = torch.load(file_name)
    return params


filename = "/home/anaelle/Workspace/procgen_adventure/params_.pkl"
filename = "/home/anaelle/Workspace/procgen_adventure/experiments/20200310/dqn_coinrun/run_0/params.pkl"

params = load_params(filename)
agent = ProcgenDqnAgent(initial_model_state_dict=params["agent_state_dict"]["model"])
config = dict(
    env=dict(
        id="procgen:procgen-coinrun-v0",
        num_levels=1,
        start_level=32,
        distribution_mode="easy",
        paint_vel_info=True,
    )
)
env = make(**config["env"])
agent.initialize(env.spaces)
collector = EvalCollector(env, agent)
collector.collect_evaluation(params["itr"])

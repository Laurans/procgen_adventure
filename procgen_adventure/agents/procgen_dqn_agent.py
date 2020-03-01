from procgen_adventure.models.dqn_model import DqnModel
from rlpyt.agents.dqn.dqn_agent import DqnAgent


class ProcgenMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            image_shape=env_spaces.observation.shape, output_size=env_spaces.action.n
        )


class ProcgenDqnAgent(ProcgenMixin, DqnAgent):
    def __init__(self, ModelCls=DqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

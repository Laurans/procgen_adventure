import torch
import torch.nn as nn

from procgen_adventure.network.utils import BaseNet, layer_init


class CategoricalActorCriticPolicy(nn.Module, BaseNet):
    def __init__(
        self,
        CHW_shape: tuple,
        action_dim: int,
        phi_body: nn.Module,
        actor_body: nn.Module,
        critic_body: nn.Module,
    ):
        super(CategoricalActorCriticPolicy, self).__init__()

        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body

        self.fc_action = layer_init(
            nn.Linear(self.actor_body.feature_dim, action_dim), w_scale=1e-3
        )
        self.fc_critic = layer_init(
            nn.Linear(self.critic_body.feature_dim, 1), w_scale=1e-3
        )

        # self.actor_params = list(self.actor_body.parameters()) + list(
        #     self.fc_action.parameters()
        # )

        # self.critic_params = list(self.critic_body.parameters()) + list(
        #     self.fc_critic.parameters()
        # )

        # self.phi_params = list(self.phi_body.parameters())

    def forward(self, obs, action=None):
        phi = self.phi_body(obs)
        phi_action = self.actor_body(phi)
        phi_value = self.critic_body(phi)

        logits = self.fc_action(phi_action)
        value = self.fc_critic(phi_value).squeeze()

        distribution = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return {
            "action": action,
            "neg_log_prob_a": -log_prob,
            "entropy": entropy,
            "value": value,
        }

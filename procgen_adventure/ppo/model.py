import collections

import torch

from procgen_adventure.network.bodies import body_factory
from procgen_adventure.network.heads import CategoricalActorCriticPolicy
from procgen_adventure.utils.torch_utils import sync_gradients, tensor, to_np


class Model:
    def __init__(
        self,
        ob_shape: tuple,
        ac_space: int,
        policy_network_archi,
        ent_coef,
        vf_coef,
        l2_coef,
        max_grad_norm,
        device,
    ):
        phi_body = body_factory(policy_network_archi)(CHW_shape=ob_shape[::-1])
        actor_body = body_factory("DummyBody")(phi_body.feature_dim)
        critic_body = body_factory("DummyBody")(phi_body.feature_dim)

        self.network = CategoricalActorCriticPolicy(
            CHW_shape=ob_shape,
            action_dim=ac_space,
            phi_body=phi_body,
            actor_body=actor_body,
            critic_body=critic_body,
        )
        self.network.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.network.parameters(), eps=1e-5)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.l2_coef = l2_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.network.forward

    def eval(self):
        self.network.eval()

    def compute_loss(
        self, obs, actions, values, returns, neglogpac_old, advs, cliprange
    ):

        prediction = self.network(obs=obs, action=actions)

        neglogpac = prediction["neg_log_prob_a"]
        policy_entropy = prediction["entropy"].mean()
        vpred = prediction["value"]
        vpredclipped = values + (vpred - values).clamp(-cliprange, cliprange)

        vf_losses1 = (vpred - returns).pow(2)
        vf_losses2 = (vpredclipped - returns).pow(2)
        value_loss = 0.5 * (torch.max(vf_losses1, vf_losses2)).mean()

        ratio = (neglogpac_old - neglogpac).exp()
        pg_losses1 = -advs * ratio
        pg_losses2 = -advs * ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
        policy_loss = (torch.max(pg_losses1, pg_losses2)).mean()

        approxkl = 0.5 * ((neglogpac - neglogpac_old).pow(2)).mean()
        clipfrac = (torch.abs(ratio - 1.0) > cliprange).float().mean()

        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.network.parameters():
            l2_reg += torch.norm(param)

        loss = policy_loss - policy_entropy * self.ent_coef + value_loss * self.vf_coef
        return (
            loss,
            collections.OrderedDict(
                policy_loss=to_np(policy_loss),
                value_loss=to_np(value_loss),
                policy_entropy=to_np(policy_entropy),
                approxkl=to_np(approxkl),
                clipfrac=to_np(clipfrac),
            ),
        )

    def train(self, lr: float, cliprange, batch: dict):
        self.network.train()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        obs = tensor(batch["states"], self.device)
        actions = tensor(batch["actions"], self.device)
        neglogpac_old = tensor(batch["neg_log_prob_a"], self.device)
        returns = tensor(batch["returns"], self.device)
        values = tensor(batch["values"], self.device)

        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        self.optimizer.zero_grad()

        loss, opt_dict = self.compute_loss(
            obs, actions, values, returns, neglogpac_old, advs, cliprange
        )

        loss.backward()
        sync_gradients(self.network)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return opt_dict

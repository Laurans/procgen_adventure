import torch

from procgen_adventure.network.bodies import body_factory
from procgen_adventure.network.heads import VanillaPolicy, DuelingNetPolicy, BaseNet
from procgen_adventure.utils.torch_utils import (
    to_np,
    sync_gradients,
    tensor,
    range_tensor,
)
import collections

import copy


class Model:
    def __init__(
        self,
        ob_shape: tuple,
        ac_space: int,
        policy_network_archi: str,
        dueling: bool,
        batch_size: int,
        max_grad_norm: float,
        sgd_update_frequency: int,
        target_network_update_freq: int,
        double_q: bool,
        discount: float,
        device: str,
    ):
        phi_body = body_factory(policy_network_archi)(CHW_shape=ob_shape[::-1])

        if dueling:
            self.network: BaseNet = DuelingNetPolicy(action_dim=ac_space, body=phi_body)
        else:
            self.network: BaseNet = VanillaPolicy(action_dim=ac_space, body=phi_body)

        self.target_network = copy.deepcopy(self.network)
        self.network.to(device)
        self.target_network.to(device)
        self.device = device

        self.optimizer = torch.optim.RMSprop(
            self.network.parameters(), lr=0.00025, alpha=0.95, ep=0.01, centered=True
        )
        self.max_grad_norm = max_grad_norm
        self.sgd_update_frequency = sgd_update_frequency
        self.target_network_update_freq = target_network_update_freq
        self.double_q = double_q
        self.discount = discount

        self.batch_indices = range_tensor(batch_size, device)
        self.total_steps = 0

        self.step = self.network.forward

    def eval(self):
        self.network.eval()

    def compute_loss(self, obs, actions, rewards, next_obs, terminals):
        q_next = self.network(obs).detach()
        if self.double_q:
            best_actions = torch.argmax(self.network(next_obs), dim=-1)
            q_next = q_next[self.batch_indices, best_actions]
        else:
            q_next = q_next.max(1)[0]

        q_next = self.discount * q_next * (1 - terminals)
        q_next.add_(rewards)
        q = self.network(obs)
        q = q[self.batch_indices, actions]
        loss = (q_next - q).pow(2).mul(0.5).mean()

        return loss, collections.OrderedDict(loss=to_np(loss))

    def train(self, lr: float, batch: dict):
        self.network.train()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        obs = tensor(batch["states"], self.device)
        actions = tensor(batch["actions"], self.device).long()
        rewards = tensor(batch["rewards"], self.device)
        next_obs = tensor(batch["next_states"], self.device)
        terminals = tensor(batch["dones"], self.device)

        self.optimizer.zero_grad()

        loss, opt_dict = self.compute_loss(obs, actions, rewards, next_obs, terminals)

        loss.backward()
        sync_gradients(self.network)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if (
            self.total_steps
            / self.sgd_update_frequency
            % self.target_network_update_freq
            == 0
        ):
            self.target_network.load_state_dict(self.network.state_dict())

        return opt_dict

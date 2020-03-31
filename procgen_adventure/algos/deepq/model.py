import collections
import copy

import numpy as np
import torch

from procgen_adventure.network.bodies import body_factory
from procgen_adventure.network.heads import BaseNet, DuelingNetPolicy, VanillaPolicy
from procgen_adventure.utils.torch_utils import (
    range_tensor,
    sync_gradients,
    tensor,
    to_np,
)


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
        n_step_return: int,
        use_prioritized_replay: bool,
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
            self.network.parameters(), lr=0.00025, alpha=0.95, eps=0.01, centered=True
        )
        self.max_grad_norm = max_grad_norm
        self.sgd_update_frequency = sgd_update_frequency
        self.target_network_update_freq = target_network_update_freq
        self.double_q = double_q
        self.discount = discount
        self.n_step_return = n_step_return
        self.use_prioritized_replay = use_prioritized_replay

        self.batch_indices = range_tensor(batch_size, device)
        self.update_counter = 0

        self.step = self.network.forward

    def eval(self):
        self.network.eval()

    def compute_loss(self, obs, actions, rewards, next_obs, terminals, weights):
        q = self.network(obs)["value"]
        q = q[self.batch_indices, actions]

        with torch.no_grad():
            q_next = self.target_network(next_obs)["value"]
            if self.double_q:
                best_actions = torch.argmax(self.network(next_obs)["value"], dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]

        q_next = (
            rewards
            + terminals.logical_not() * (self.discount ** self.n_step_return) * q_next
        )
        delta = q_next - q
        losses = delta.pow(2).mul(0.5)

        if self.use_prioritized_replay:
            losses *= weights

        loss = losses.mean()
        tb_abs_errors = abs(delta).detach()

        return (
            loss,
            collections.OrderedDict(loss=to_np(losses), tbAbsErr=to_np(tb_abs_errors)),
        )

    def _preprocess_batch(self, batch):
        obs = tensor(batch["states"], self.device)
        actions = tensor(batch["actions"], self.device).long()
        if len(actions.shape) == 2:
            actions = actions.squeeze(-1)
        rewards = tensor(batch["rewards"], self.device)
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(-1)
        next_obs = tensor(batch["next_states"], self.device)
        terminals = tensor(batch["dones"], self.device)
        if len(terminals.shape) == 2:
            terminals = terminals.squeeze(-1)
        weights = (
            tensor(batch["weights"], self.device)
            if batch["weights"] is not None
            else None
        )

        return obs, actions, rewards, next_obs, terminals, weights

    def train(self, lr: float, batch: dict):
        self.network.train()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        preprocessed_batch = self._preprocess_batch(batch)

        self.optimizer.zero_grad()

        loss, opt_dict = self.compute_loss(*preprocessed_batch)

        loss.backward()
        sync_gradients(self.network)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        self.update_counter += 1

        if self.update_counter % self.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        opt_dict["gradNorm"] = np.array([grad_norm])
        return opt_dict

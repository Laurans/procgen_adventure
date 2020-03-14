import datetime
import time
from collections import defaultdict, deque
from typing import Dict, List

from procgen_adventure.algos.base import Algo
from procgen_adventure.algos.deepq.model import Model
from procgen_adventure.algos.utils import get_values_from_list_dict
from procgen_adventure.utils.logger import MyLogger
from procgen_adventure.utils.torch_utils import (
    sync_initial_weights,
    sync_values,
    tensor,
    to_np,
)
from procgen_adventure.algos.deepq.sampler import Sampler
from .config import DeepQExpConfig
from .utils import get_example_outputs

from procgen_adventure.replays.non_sequence.frame import (
    UniformReplayFrameBuffer,
    PrioritizedReplayFrameBuffer,
    AsyncUniformReplayFrameBuffer,
    AsyncPrioritizedReplayFrameBuffer,
)


def get_model(env, config, device):
    model = Model(
        ob_shape=env.observation_space.shape,
        ac_space=env.action_space.n,
        policy_network_archi=config.ARCHITECTURE,
        dueling=config.IS_DUELING_NETWORK,
        batch_size=config.BATCH_SIZE,
        max_grad_norm=config.MAX_GRAD_NORM,
        sgd_update_frequency=config.SGD_UPDATE_FREQ,
        target_network_update_freq=config.TARGET_UPDATE_FREQ,
        double_q=config.IS_DOUBLE_Q,
        discount=config.DISCOUNT,
        device=device,
    )

    return model


class DeepQ(Algo):
    def __init__(self, rank, env, config: DeepQExpConfig, logger: MyLogger):
        self.rank = rank
        self.config: DeepQExpConfig = config
        self.env = env
        self.logger: MyLogger = logger
        self.device = f"{self.config.DEVICE}:{rank}"
        self.model: Model = get_model(env, config, self.device)
        sync_initial_weights(self.model.network)
        self.logger.set_snapshot_mode("gap")
        self.logger.set_snapshot_gap(self.config.SAVE_INTERVAL)
        self.initialize_replay_buffer()

    def initialize_replay_buffer(self, async_=False):
        example_to_buffer = get_example_outputs(self.env)

        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.config.REPLAY_SIZE,
            B=self.config.NUM_ENVS,
            discount=self.config.DISCOUNT,
            n_step_return=self.config.N_STEP_RETURN,
        )

        if self.config.PRIORITIZED_REPLAY:
            replay_kwargs.update(
                dict(alpha=self.config.PRI_ALPHA, beta=self.config.PRI_BETA_INIT)
            )
            ReplayCls = (
                AsyncPrioritizedReplayFrameBuffer
                if async_
                else PrioritizedReplayFrameBuffer
            )
        else:
            ReplayCls = (
                AsyncUniformReplayFrameBuffer if async_ else UniformReplayFrameBuffer
            )

        self.replay_buffer = ReplayCls(**replay_kwargs)

    # def optim_initialize(self):
    #    self.pri_beta_itr = max(1, self.config.PRI_BETA_STEPS // self.env.num_envs)

    def get_itr_snapshot(self, update):
        return (
            {
                "update": update,
                "model_state_dict": self.model.network.state_dict(),
                "optimizer_state_dict": self.model.optimizer.state_dict(),
            },
        )

    def _log_diagnostics(self, update, time_elapsed, fps, lossvals):
        if update % self.config.LOG_INTERVAL == 0 or update == 1:
            rew_100 = get_values_from_list_dict(self.epinfobuf100, "r")
            rew_10 = get_values_from_list_dict(self.epinfobuf10, "r")
            ep_len = get_values_from_list_dict(self.epinfobuf100, "l")

            completion_perc = update * self.config.NBATCH / self.config.TOTAL_TIMESTEPS
            time_remaining = datetime.timedelta(
                seconds=(time_elapsed / completion_perc - time_elapsed)
            )

            self.logger.record_tabular("misc/iter_update", update)
            self.logger.record_tabular(
                "misc/total_timesteps", update * self.config.NBATCH
            )
            self.logger.record_tabular("fps", fps)
            self.logger.record_tabular("misc/time_elapsed", time_elapsed)
            self.logger.record_tabular_misc_stat("episode/length_100/", ep_len)
            self.logger.record_tabular_misc_stat("episode/rew_100/", rew_100)
            self.logger.record_tabular_misc_stat("episode/rew_10/", rew_10)
            self.logger.record_tabular("misc/completion_training", completion_perc)
            self.logger.log(f"Time remaining {time_remaining}")

            for lossname, lossval, in lossvals.items():
                self.logger.record_tabular_misc_stat(f"loss/{lossname}/", lossval)

            self.logger.dump_tabular()

    def learn(self):
        sampler = Sampler(
            self.env,
            self.model,
            self.replay_buffer,
            self.device,
            self.config.EXPLORATION_STEPS,
            self.config.RANDOM_ACTION_PROB_FN,
            self.config.BATCH_SIZE,
        )

        sampler.run_exploration_steps()
        sampler.run()

        """
        self.epinfobuf10 = deque(maxlen=10)
        self.epinfobuf100 = deque(maxlen=100)

        tfirststart = time.perf_counter()

        nupdates = int(self.config.TOTAL_TIMESTEPS // self.config.NBATCH)

        for update in range(1, nupdates + 1):
            if self.rank == 0:
                self.logger.log(f"{update}/{nupdates+1}")

            # Start timer
            tstart = time.perf_counter()

            lossvals, epinfos = run_update(
                update, nupdates, sampler, self.model, self.config
            )

            self.epinfobuf10.extend(epinfos)
            self.epinfobuf100.extend(epinfos)

            # End timer
            tnow = time.perf_counter()
            time_elapsed = tnow - tfirststart

            # Calculate the fps
            fps = int(self.config.NBATCH / (tnow - tstart))

            self._log_diagnostics(update, time_elapsed, fps, lossvals)
            self.logger.save_itr_params(update, self.get_itr_snapshot(update))

        self.logger.save_itr_params(update, self.get_itr_snapshot(update), force=True)

        self.env.close()
        """


def run_update(update: int, nupdates: int, sampler, model: Model, Config):
    # Calculate the learning rate
    lrnow = Config.LR_FN(update)

    # Get minibatch
    data_sampled, epinfos = sampler.run()

    # For each minibatch we'll calculate the loss and append it
    mblossvals: Dict[str, List] = defaultdict(list)
    for _ in range(Config.NUM_OPT_EPOCHS):
        slices = {key: data_sampled[key][mbinds] for key in data_sampled}
        losses = model.train(lrnow, slices)
        for key, val in losses.items():
            mblossvals[key].append(val)

    return mblossvals, epinfos

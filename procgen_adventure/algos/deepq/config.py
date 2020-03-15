from procgen_adventure.utils.scheduler import ConstantSchedule, LinearSchedule


class DeepQExpConfig:
    def __init__(self):
        self.PROJECT = ""
        self.TAGS = ["DeepQ"]

        self.WORLD_SIZE = 1
        self.DEVICE = "cuda"

        self.NUM_ENVS = 8  # 32 * (8 // self.WORLD_SIZE)
        self.TOTAL_TIMESTEPS = 256e6

        self.REPLAY_SIZE = int(1e5)
        self.PRIORITIZED_REPLAY = False
        self.PRI_ALPHA = 0.6
        self.PRI_BETA_INIT = 0.4
        self.PRI_BETA_STEPS = int(50e6)

        self.N_STEP_RETURN = 3
        self.BATCH_SIZE = 32

        self.LEARNING_RATE = 0.0000625  # Learning rate, constant 0.00025 / 4
        self.LR_FN = ConstantSchedule(self.LEARNING_RATE)
        self.SGD_UPDATE_FREQ = 4
        self.NBATCH = self.NUM_ENVS * self.SGD_UPDATE_FREQ
        self.TARGET_UPDATE_FREQ = 8000

        self.EXPLORATION_STEPS = 20000
        self.RANDOM_ACTION_PROB_FN = LinearSchedule(1.0, 0.01, 1e6)
        self.MAX_GRAD_NORM = 10  # Gradient norm clipping coefficient
        self.DISCOUNT = 0.99  # discounting factor
        self.NUM_OPT_EPOCHS = 1  # number of training epochs per update

        self.LOG_INTERVAL = 300  # Number of updates betwen logging events

        self.IS_DUELING_NETWORK = True
        self.IS_DOUBLE_Q = True

        # The convolutional architecture to use
        # One of {'NatureConv', 'impala', 'impalalarge'}
        self.ARCHITECTURE = "NatureConv"

        # Should the model include an LSTM
        self.USE_LSTM = False

        # Should batch normalization be used after each convolutional layer
        # NOTE: Only applies to IMPALA and IMPALA-Large architectures
        self.USE_BATCH_NORM = True

        # What dropout probability to use after each convolutional layer
        # NOTE: Only applies to IMPALA and IMPALA-Large architectures
        self.DROPOUT = 0.0

        # The L2 penalty to use during training
        self.L2_WEIGHT = 1e-4

        # The number of frames to stack for each observation.
        self.FRAME_STACK = 1

        # Overwrite the latest save file after this many updates
        self.SAVE_INTERVAL = 300

        self.ENV_CONFIG = dict(
            env_name="coinrun",
            num_levels=1,
            start_level=32,
            distribution_mode="easy",
            paint_vel_info=True,
            num_envs=self.NUM_ENVS,
        )

        self.TAGS += [self.ARCHITECTURE]

    def to_config_dict(self):
        config = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, str, bool, tuple, list]:
                config[k] = v
        return config

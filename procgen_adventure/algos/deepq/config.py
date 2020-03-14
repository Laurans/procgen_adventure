from procgen_adventure.utils.scheduler import ConstantSchedule


class DeepQExpConfig:
    def __init__(self):
        self.PROJECT = ""
        self.TAGS = ["DeepQ"]

        self.WORLD_SIZE = 1
        self.DEVICE = "cuda"

        self.NUM_ENVS = 2  # 32 * (8 // self.WORLD_SIZE)
        self.TOTAL_TIMESTEPS = 256e6

        self.LEARNING_RATE = 5e-4  # Learning rate, constant
        self.LR_FN = ConstantSchedule(self.LEARNING_RATE)
        self.MAX_GRAD_NORM = 10  # Gradient norm clipping coefficient
        self.GAMMA = 0.999  # discounting factor
        self.NUM_OPT_EPOCHS = 1  # number of training epochs per update

        self.LOG_INTERVAL = 5  # Number of updates betwen logging events

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

        # The probability the agent's action is replaced with a random action
        self.EPSILON_GREEDY = 0.0

        # The number of frames to stack for each observation.
        self.FRAME_STACK = 1

        # Overwrite the latest save file after this many updates
        self.SAVE_INTERVAL = 10

        self.ENV_CONFIG = dict(
            env_name="coinrun",
            num_levels=1,
            start_level=32,
            distribution_mode="easy",
            paint_vel_info=True,
            num_envs=self.NUM_ENVS,
        )

        self.BATCH_SIZE = 32

        self.TAGS += [self.ARCHITECTURE]

    def to_config_dict(self):
        config = {}
        for k, v in self.__dict__.items():
            if type(v) in [int, float, str, bool, tuple, list]:
                config[k] = v
        return config

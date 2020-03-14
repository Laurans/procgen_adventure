def get_alg_class():
    from .entrypoint import PPO

    return PPO


def get_alg_config():
    from .config import PPOExpConfig

    return PPOExpConfig

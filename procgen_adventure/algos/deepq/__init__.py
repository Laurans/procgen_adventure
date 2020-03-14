def get_alg_class():
    from .entrypoint import DeepQ

    return DeepQ


def get_alg_config():
    from .config import DeepQExpConfig

    return DeepQExpConfig

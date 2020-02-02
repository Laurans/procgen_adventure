import ray
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer

config = DEFAULT_CONFIG.copy()
config["use_pytorch"] = True


ray.init()
trainer = PPOTrainer(env="procgen:procgen-coinrun-v0", config=config)

import gymnasium as gym
import highway_env
import pprint

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 2
}

env = gym.make("highway-v0", config=config)
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Observation space: {env.observation_space}")
env.close()

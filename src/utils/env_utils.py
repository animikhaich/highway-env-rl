import gymnasium as gym
import highway_env
import warnings

def make_env(scenario, render_mode=None):
    """
    Creates and configures the highway-env environment.

    We use GrayscaleObservation with frame stacking to provide temporal context to the agent
    while keeping the input dimensionality manageable (128x64).

    Args:
        scenario (str): Name of the scenario ("highway", "merge", etc.).
        render_mode (str, optional): Rendering mode ("rgb_array", "human", or None).

    Returns:
        gym.Env: The configured environment.
    """

    env_id_map = {
        "highway": "highway-v0",
        "merge": "merge-v0",
        "roundabout": "roundabout-v0",
        "parking": "parking-v0",
        "intersection": "intersection-v0",
        "racetrack": "racetrack-v0"
    }

    if scenario not in env_id_map:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(env_id_map.keys())}")

    env_id = env_id_map[scenario]

    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 500, # Max episode duration
    }

    # render_mode argument is passed to gym.make
    env = gym.make(env_id, render_mode=render_mode, config=config)
    env.unwrapped.configure(config)

    return env

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.env_utils import make_env
from src.agents.dqn import DQN
from src.agents.ppo import PPO


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    render_mode = "rgb_array" if args.render else None
    env = make_env(args.scenario, render_mode=render_mode)

    # Initialize agent
    config = {
        "model_name": args.model,
        "model_output_dim": args.output_dim,
        # Dummy values for evaluation
        "lr": 0,
        "gamma": 0,
        "batch_size": 0,
        "target_update": 0,
        "epsilon_start": 0,
        "epsilon_end": 0,
        "epsilon_decay": 1,
    }

    if args.model == "custom":
        config["layers_config"] = [
            {"out_channels": 16, "kernel_size": 5, "stride": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 2},
        ]

    if args.algo == "dqn":
        agent = DQN(env.observation_space, env.action_space, config, device)
    elif args.algo == "ppo":
        agent = PPO(env.observation_space, env.action_space, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Load model
    if not os.path.exists(args.load_path):
        raise ValueError(f"Model file not found: {args.load_path}")

    agent.load(args.load_path)
    print(f"Model loaded from {args.load_path}")

    # Print model and settings information
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Model: {args.model}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.episodes}")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    print()

    rewards = []

    for ep in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        frames = []

        while not (done or truncated):
            if args.render:
                frame = env.render()
                frames.append(frame)

            action = agent.select_action(state, evaluate=True)

            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {ep+1}: Reward {episode_reward}")

        if args.render and len(frames) > 0:
            # Save video
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

                clip = ImageSequenceClip([f for f in frames], fps=30)
                os.makedirs("videos", exist_ok=True)
                video_path = f"videos/{args.algo}_{args.scenario}_ep{ep+1}.mp4"
                clip.write_videofile(video_path, logger=None)
                print(f"Video saved to {video_path}")
            except ImportError:
                print("moviepy not installed, cannot save video.")
            except Exception as e:
                print(f"Error saving video: {e}")

    avg_reward = np.mean(rewards)
    print(f"Average Reward over {args.episodes} episodes: {avg_reward}")
    return avg_reward

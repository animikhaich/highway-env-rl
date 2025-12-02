import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.env_utils import make_env
from src.agents.dqn import DQN
from src.agents.ppo import PPO
import gc


def train(args):
    """
    Main training loop for the RL agent.

    Args:
        args: Parsed command line arguments containing training configuration.
              Includes algo, model, scenario, total_timesteps, etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    # We use "rgb_array" for rendering if requested. This avoids opening a window
    # but allows capturing frames if needed (though currently we just discard them
    # in the loop, potentially for monitoring or future video saving).
    render_mode = "rgb_array" if args.render else None
    env = make_env(args.scenario, render_mode=render_mode)

    # Initialize agent configuration
    # These hyperparameters are chosen based on standard DQN/PPO baselines for highway-env.
    config = {
        "model_name": args.model,
        "model_output_dim": args.output_dim,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": args.total_timesteps * 0.1,  # Decay epsilon over 10% of total steps
        "memory_size": 15000,  # Replay buffer size
        "target_update": 50,   # Update target network every 50 steps
        "k_epochs": 4,         # PPO optimization epochs
        "eps_clip": 0.2,       # PPO clipping parameter
    }

    # Custom layers config if model is custom
    if args.model == "custom":
        # Example custom config
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

    # Print model and settings information
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Model: {args.model}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Learning rate: {args.lr}")
    print(f"Discount factor (gamma): {args.gamma}")
    print(f"Batch size: {args.batch_size}")
    if args.algo == "dqn":
        print(f"Epsilon start: {config['epsilon_start']}")
        print(f"Epsilon end: {config['epsilon_end']}")
        print(f"Epsilon decay: {config['epsilon_decay']}")
        print(f"Memory size: {config['memory_size']}")
        print(f"Target update frequency: {config['target_update']}")
    elif args.algo == "ppo":
        print(f"K epochs: {config['k_epochs']}")
        print(f"Epsilon clip: {config['eps_clip']}")
    if args.model == "custom":
        print(f"Custom layers config: {config['layers_config']}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Scenario: {args.scenario}")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    print()

    # Training Loop
    episode_rewards = []
    episode_lengths = []
    # losses list removed to avoid potential memory accumulation if loss is not properly detached
    # Although loss.item() returns a float, accumulating 10k+ items is unnecessary if not used.

    timestep = 0
    episode_idx = 0

    pbar = tqdm(total=args.total_timesteps)

    while timestep < args.total_timesteps:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False

        while not (done or truncated):
            if args.render:
                env.render()

            if args.algo == "dqn":
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                agent.update()

                if timestep % agent.target_update == 0:
                    agent.update_target_network()
            elif args.algo == "ppo":
                action, log_prob = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                agent.store_transition((state, action, log_prob, reward, done))
                # PPO update is usually done after some steps or at end of episode.
                pass

            state = next_state
            episode_reward += reward
            episode_length += 1
            timestep += 1
            pbar.update(1)

            if timestep >= args.total_timesteps:
                break

        # PPO Update at end of episode
        if args.algo == "ppo":
            agent.update()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_idx += 1
        
        if episode_idx % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            pbar.set_description(f"Ep: {episode_idx}, Avg Reward: {avg_reward:.2f}")

    pbar.close()
    env.close()

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(
        args.save_dir, f"{args.algo}_{args.scenario}_{args.model}.pth"
    )
    agent.save(save_path)
    print(f"Model saved to {save_path}")

    # Plot results
    plt.figure()
    plt.plot(episode_rewards)
    plt.title(f"{args.algo} on {args.scenario}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(
        os.path.join(args.save_dir, f"training_curve_{args.algo}_{args.scenario}.png")
    )
    print(f"Training curve saved.")

    return save_path

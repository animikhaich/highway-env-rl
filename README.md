# HighwayEnv RL Project

This project implements Reinforcement Learning (RL) agents to solve various autonomous driving scenarios in the [HighwayEnv](https://github.com/eleurent/highway-env) environment. It features a modular codebase with support for DQN and PPO algorithms, customizable Convolutional Neural Networks (CNNs), and different driving scenarios.

## Features

- **Algorithms**: 
  - **DQN** (Deep Q-Network) with Target Network and Replay Buffer.
  - **PPO** (Proximal Policy Optimization) with Actor-Critic architecture.
- **Models**:
  - **SmallCNN**: A lightweight LeNet-like architecture.
  - **LargeCNN**: A deeper network for more complex feature extraction.
  - **CustomCNN**: Fully customizable via configuration.
- **Scenarios**: Supports standard HighwayEnv scenarios like `highway`, `merge`, `roundabout`, `parking`, `intersection`, `racetrack`.
- **Visualization**: Live training plots (saved to disk) and video rendering during evaluation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install highway-env gymnasium torch matplotlib numpy pygame moviepy
   ```

## Usage

The project is controlled via `main.py` with command-line arguments.

### Training

To train an agent, use `--mode train`.

**Example: Train DQN on Highway scenario**
```bash
python main.py --mode train --algo dqn --scenario highway --model small --total_timesteps 10000 --batch_size 32
```

**Arguments:**
- `--algo`: `dqn` or `ppo`.
- `--scenario`: `highway`, `merge`, `roundabout`, `parking`, `intersection`, `racetrack`.
- `--model`: `small`, `large`, `custom`.
- `--total_timesteps`: Total number of environmental steps to train.
- `--batch_size`: Batch size for updates.
- `--lr`: Learning rate (default: 1e-4).
- `--gpu`: Add this flag to use GPU if available.
- `--save_dir`: Directory to save checkpoints and plots (default: `checkpoints`).

### Evaluation

To evaluate a trained agent and see it in action (or save videos), use `--mode eval`.

**Example: Evaluate the trained DQN model**
```bash
python main.py --mode eval --algo dqn --scenario highway --model small --load_path checkpoints/dqn_highway_small.pth --episodes 5 --render
```

**Arguments:**
- `--load_path`: Path to the `.pth` model file.
- `--episodes`: Number of episodes to run.
- `--render`: Enable rendering. This will save videos to the `videos/` directory if `moviepy` is installed.

## Project Structure

```
.
├── main.py                 # Entry point
├── src/
│   ├── agents/             # RL Agent implementations
│   │   ├── base_agent.py
│   │   ├── dqn.py
│   │   └── ppo.py
│   ├── models/             # Neural Network architectures
│   │   └── cnn.py
│   ├── utils/              # Utilities
│   │   ├── env_utils.py    # Environment wrapping
│   │   └── plotter.py
│   ├── train.py            # Training loop
│   └── evaluate.py         # Evaluation loop
├── checkpoints/            # Saved models and training curves
├── videos/                 # Recorded videos of evaluation
└── assets/                 # Documentation assets
```

## Customization

### Adding a Custom Model
You can define a custom CNN architecture by modifying the `config` dictionary in `train.py` or by passing JSON-like config if extended. Currently, `CustomCNN` takes a list of layer configurations.

### Hyperparameters
Hyperparameters for DQN and PPO are defined in their respective classes (`src/agents/dqn.py`, `src/agents/ppo.py`) or passed via the `config` dictionary in `train.py`.


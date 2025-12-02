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
  - **ResNetCNN**: ResNet-style architecture with residual blocks for improved gradient flow.
  - **VGGCNN**: VGG-style deep convolutional network with extensive feature extraction.
- **Scenarios**: Supports standard HighwayEnv scenarios like `highway`, `merge`, `roundabout`, `parking`, `intersection`, `racetrack`.
- **Visualization**: Live rendering during training/evaluation, training plots (saved to disk), and video rendering during evaluation.
- **Configuration Output**: Prints selected model and hyperparameters at startup for verification.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes all necessary packages: PyTorch, Gymnasium, HighwayEnv, NumPy, Matplotlib, tqdm, and moviepy (optional for video saving).

## Usage

The project is controlled via `main.py` with command-line arguments.

### Training

To train an agent, use `--mode train`.

#### Example: Train DQN on Highway scenario with ResNet

```bash
python main.py --mode train --algo dqn --scenario highway --model resnet --total_timesteps 10000 --batch_size 32 --gpu
```

**Arguments:**

- `--algo`: `dqn` or `ppo`.
- `--scenario`: `highway`, `merge`, `roundabout`, `parking`, `intersection`, `racetrack`.
- `--model`: `small`, `large`, `custom`, `resnet`, `vgg`.
- `--total_timesteps`: Total number of environmental steps to train.
- `--batch_size`: Batch size for updates.
- `--lr`: Learning rate (default: 1e-4).
- `--gamma`: Discount factor (default: 0.99).
- `--output_dim`: Output dimension of the CNN feature extractor (default: 512).
- `--gpu`: Add this flag to use GPU if available.
- `--render`: Enable rendering during training (shows environment window).
- `--save_dir`: Directory to save checkpoints and plots (default: `checkpoints`).

### Evaluation

To evaluate a trained agent and see it in action (or save videos), use `--mode eval`.

#### Example: Evaluate the trained DQN model

```bash
python main.py --mode eval --algo dqn --scenario highway --model resnet --load_path checkpoints/dqn_highway_resnet.pth --episodes 5 --render
```

**Arguments:**

- `--load_path`: Path to the `.pth` model file.
- `--episodes`: Number of episodes to run.
- `--render`: Enable rendering. This will display the environment and save videos to the `videos/` directory if `moviepy` is installed.

## Project Structure

```
.
├── main.py                 # Entry point with argument parsing
├── check_env.py            # Environment checking script
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file
├── README.md               # This file
├── src/
│   ├── agents/             # RL Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py   # Base agent class
│   │   ├── dqn.py          # DQN implementation
│   │   └── ppo.py          # PPO implementation
│   ├── models/             # Neural Network architectures
│   │   ├── __init__.py
│   │   └── cnn.py          # CNN model definitions
│   ├── utils/              # Utilities
│   │   ├── __init__.py
│   │   ├── env_utils.py    # Environment setup and configuration
│   │   └── plotter.py      # Plotting utilities
│   ├── train.py            # Training loop
│   └── evaluate.py         # Evaluation loop
├── checkpoints/            # Saved models and training curves (created during training)
├── videos/                 # Recorded videos of evaluation (created during evaluation with --render)
└── assets/                 # Documentation assets
```

## Customization

### Adding a Custom Model

You can define a custom CNN architecture by modifying the `config` dictionary in `train.py` or by extending the `get_model` function in `src/models/cnn.py`. The `CustomCNN` takes a list of layer configurations.

### Hyperparameters

Hyperparameters for DQN and PPO are defined in their respective classes (`src/agents/dqn.py`, `src/agents/ppo.py`) or passed via the `config` dictionary in `train.py`. The training script prints all relevant hyperparameters at startup for easy verification.

### Environment Configuration

The environment configuration is handled in `src/utils/env_utils.py`. You can modify observation types, action spaces, and other settings there.


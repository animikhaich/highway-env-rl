import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.agents.base_agent import BaseAgent
from src.models.cnn import get_model


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(BaseAgent):
    def __init__(self, observation_space, action_space, config, device):
        super(DQN, self).__init__(observation_space, action_space, device)
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 1000)
        self.target_update = config.get("target_update", 10)
        self.memory = ReplayBuffer(config.get("memory_size", 10000))

        input_shape = observation_space.shape
        self.n_actions = action_space.n

        model_name = config.get("model_name", "small")
        model_output_dim = config.get("model_output_dim", 512)

        # Policy Network
        # Remove model_name from kwargs because it is already passed as positional argument
        if "model_name" in config:
            del config["model_name"]
        self.feature_extractor = get_model(
            model_name, input_shape, model_output_dim, **config
        ).to(device)
        self.head = nn.Linear(model_output_dim, self.n_actions).to(device)

        # Target Network
        self.target_feature_extractor = get_model(
            model_name, input_shape, model_output_dim, **config
        ).to(device)
        self.target_head = nn.Linear(model_output_dim, self.n_actions).to(device)
        self.target_feature_extractor.load_state_dict(
            self.feature_extractor.state_dict()
        )
        self.target_head.load_state_dict(self.head.state_dict())
        self.target_feature_extractor.eval()
        self.target_head.eval()

        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.head.parameters()),
            lr=config.get("lr", 1e-4),
        )

        self.steps_done = 0

    def select_action(self, state, evaluate=False):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

        if not evaluate:
            self.steps_done += 1

        if evaluate or sample > eps_threshold:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                features = self.feature_extractor(state_t)
                q_values = self.head(features)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def update(self, batch=None):
        if len(self.memory) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        features = self.feature_extractor(state)
        q_values = self.head(features)
        state_action_values = q_values.gather(1, action)

        with torch.no_grad():
            next_features = self.target_feature_extractor(next_state)
            next_q_values = self.target_head(next_features)
            next_state_values = next_q_values.max(1)[0]
            expected_state_action_values = (next_state_values * self.gamma) * (
                1 - done
            ) + reward

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_feature_extractor.load_state_dict(
            self.feature_extractor.state_dict()
        )
        self.target_head.load_state_dict(self.head.state_dict())

    def save(self, path):
        torch.save(
            {
                "feature_extractor": self.feature_extractor.state_dict(),
                "head": self.head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.head.load_state_dict(checkpoint["head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.target_feature_extractor.load_state_dict(
            self.feature_extractor.state_dict()
        )
        self.target_head.load_state_dict(self.head.state_dict())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from src.agents.base_agent import BaseAgent
from src.models.cnn import get_model


class PPO(BaseAgent):
    def __init__(self, observation_space, action_space, config, device):
        super(PPO, self).__init__(observation_space, action_space, device)
        self.config = config
        self.gamma = config.get("gamma", 0.99)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.K_epochs = config.get("k_epochs", 4)
        self.lr = config.get("lr", 3e-4)

        input_shape = observation_space.shape
        self.n_actions = action_space.n

        model_name = config.get("model_name", "small")
        model_output_dim = config.get("model_output_dim", 512)

        # Shared feature extractor or separate? Usually separate is safer for stability, but shared is more efficient.
        # User asked for "give it to the real model".
        # I'll use separate feature extractors for Actor and Critic to be safe and simple.

        # Actor
        if "model_name" in config:
            del config["model_name"]
        self.actor_fe = get_model(
            model_name, input_shape, model_output_dim, **config
        ).to(device)
        self.actor_head = nn.Linear(model_output_dim, self.n_actions).to(device)

        # Critic
        self.critic_fe = get_model(
            model_name, input_shape, model_output_dim, **config
        ).to(device)
        self.critic_head = nn.Linear(model_output_dim, 1).to(device)

        self.optimizer = optim.Adam(
            [
                {"params": self.actor_fe.parameters(), "lr": self.lr},
                {"params": self.actor_head.parameters(), "lr": self.lr},
                {"params": self.critic_fe.parameters(), "lr": self.lr},
                {"params": self.critic_head.parameters(), "lr": self.lr},
            ]
        )

        self.policy_old_fe = get_model(
            model_name, input_shape, model_output_dim, **config
        ).to(device)
        self.policy_old_head = nn.Linear(model_output_dim, self.n_actions).to(device)

        self.policy_old_fe.load_state_dict(self.actor_fe.state_dict())
        self.policy_old_head.load_state_dict(self.actor_head.state_dict())

        self.buffer = []

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            features = self.policy_old_fe(state)
            action_probs = torch.softmax(self.policy_old_head(features), dim=-1)
            dist = Categorical(action_probs)

            if evaluate:
                action = torch.argmax(action_probs, dim=1)
                return action.item()

            action = dist.sample()
            action_logprob = dist.log_prob(action)

            return action.item(), action_logprob.item()

    def store_transition(self, transition):
        # state, action, log_prob, reward, done
        self.buffer.append(transition)

    def update(self, batch=None):
        # Convert buffer to tensors
        states = torch.tensor(
            np.array([t[0] for t in self.buffer]), dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            np.array([t[1] for t in self.buffer]), dtype=torch.int64
        ).to(self.device)
        log_probs_old = torch.tensor(
            np.array([t[2] for t in self.buffer]), dtype=torch.float32
        ).to(self.device)
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]

        # Monte Carlo estimate of returns
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        loss_val = 0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            features = self.actor_fe(states)
            action_probs = torch.softmax(self.actor_head(features), dim=-1)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()

            critic_features = self.critic_fe(states)
            state_values = self.critic_head(critic_features).squeeze()

            # Ratios
            ratios = torch.exp(log_probs - log_probs_old)

            # Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * nn.MSELoss()(state_values, returns)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            loss_val += loss.mean().item()

        # Copy new weights to old policy
        self.policy_old_fe.load_state_dict(self.actor_fe.state_dict())
        self.policy_old_head.load_state_dict(self.actor_head.state_dict())

        self.buffer = []
        return loss_val / self.K_epochs

    def save(self, path):
        torch.save(
            {
                "actor_fe": self.actor_fe.state_dict(),
                "actor_head": self.actor_head.state_dict(),
                "critic_fe": self.critic_fe.state_dict(),
                "critic_head": self.critic_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_fe.load_state_dict(checkpoint["actor_fe"])
        self.actor_head.load_state_dict(checkpoint["actor_head"])
        self.critic_fe.load_state_dict(checkpoint["critic_fe"])
        self.critic_head.load_state_dict(checkpoint["critic_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.policy_old_fe.load_state_dict(self.actor_fe.state_dict())
        self.policy_old_head.load_state_dict(self.actor_head.state_dict())

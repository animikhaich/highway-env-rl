from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    def __init__(self, observation_space, action_space, device):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def select_action(self, state, evaluate=False):
        pass

    @abstractmethod
    def update(self, batch):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

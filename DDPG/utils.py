import numpy as np
import torch


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        

def soft_update(target, source, tau):
    """
    update the parameters from source network to target network
    y = /tau * x + (1 - /tau)* y
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class RandomActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.1, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.x = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.x)
        dx = dx + self.sigma * np.random.rand(len(self.x))
        self.x = self.x + dx
        return self.x
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

class MLP(nn.Module):
	def __init__(self, observation_dim, action_dim):
		super(MLP, self).__init__()
		self.fc1 == nn.Linear(observation_dim, 128)
		self.fc2 == nn.Linear(128, action_dim)

	def forward(self, observation):
		x = F.relu(self.fc1(observation))
		x = F.tanh(self.fc2(x))

		return x
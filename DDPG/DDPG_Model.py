import torch
from torch import nn
from torch.nn import functional as F


class Value_Net(nn.Module):	# Critic
	def __init__(self, observation_dim, action_dim):
		super(Value_Net, self).__init__()
		self.fc1 = nn.Linear(observation_dim + action_dimn, 256)
		self.fc2 = nn.Linear(256, 512)
		self.fc3 = nn.Linear(512, 256)
		self.fc4 = nn.Linear(256, action_dim)

	def forward(self, action, observation):
		x = torch.cat((state, action), dim=1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x


class Policy_Net(nn.Module):	# Actor
	def __init__(self, observation_dim, action_dim):
		super(Policy_Net):
		self.fc1 = nn.Linear(observation_dim, 256)
		self.fc2 = nn.Linear(256, 512)
		self.fc3 = nn.Linear(512, 256)
		self.fc4 = nn.Linear(256, action_dim)

	def forward(self, observation):
		x = F.relu(self.fc1(observation))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.tanh(self.fc4(x))
		return x


class DDPG(nn.Module):
	def __init__(self, observation_dim, action_dim):
		super(DDPG, self).__init__()
		self.observation_dim = observation_dim
		self.action_dim = action_dim

		self.actor = Policy_Net(self.observation_dim, self.action_dim)
		self.critic = Value_Net(self.observation_dim, self.action_dim)

	def forward(self, state):
		action = self.actor(state)
		value = self.critic(state, action)
		return action ,value
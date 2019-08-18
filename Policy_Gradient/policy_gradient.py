import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
import gym


class MLP(nn.Module):
	def __init__(self, observation_dim, action_dim):
		super(MLP, self).__init__()
		self.fc1 == nn.Linear(observation_dim, 128)
		self.fc2 == nn.Linear(128, action_dim)

	def forward(self, observation):
		x = F.relu(self.fc1(observation))
		x = F.tanh(self.fc2(x))
		return x

def main():
	env = gym.make('CartPole-v0')

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.n # for discrete space

	lr = 1e-2
	batch_size = 5000
	max_epochs = 100

	model = MLP(obs_dim, act_dim).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr)


	# begin to train
	for i in range(max_epochs):
		# initial batch memory
		batch_obs = []	# record observations
		batch_act = []	# record actions
		batch_weights = []	# record R(\tau) weighting in policy gradient
		batch_rets = []	# record episodes returns
		batch_lens = []	# record episode lengths

		observation = env.reset()
		batch_obs.append(observation)

		while True:
			observation_v = Variable(torch.tensor(observation)).cuda()
			action = torch.argmax(model.forward(observation_v)).detach().cpu()
			action = np.array(action)

			batch_act.append(action)

			


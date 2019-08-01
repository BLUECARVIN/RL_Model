import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn

import gym
import numpy as np
import utils
import DDPG_Model


class DDPGAgent:
	def __init__(self, obs_space, action_space, ram):
		self.obs_dim = obs_space.shape[0]
		self.act_dim = action_space.shape[0] # only for continiuous env

		# just for one action 
		self.action_low = action_space.low[0]
		self.action_high = action_space.high[0]

		self.ram = ram
		self.iter = 1
		self.steps = 0
		self.gamma = 0.9
		self.batch_size = 64
		self.initial_e = 0.5
		self.end_e = 0.01
		self.e = self.initial_e

		self.start_training = 100
		self.tau = 0.01
		self.critic_lr = 0.001
		self.actor_lr = 0.001
		self.noise = utils.RandomActionNoise(self.act_dim)

		target_net = DDPG_Model.DDPG(self.obs_dim, self.act_dim).cuda()
		learning_net = DDPG_Model.DDPG(self.obs_dim, self.act_dim).cuda()
		utils.hard_update(target_net, learning_net)

		self.AC = learning_net
		self.AC_T = target_net
		self.actor = self.AC.actor
		self.critic = self.AC.critic
		self.actor_T = self.AC_T.actor
		self.critic_T = self.AC_T.critic

		self.actor_optimizer = torch.optim.Adam(self.AC.actor.parameters(), self.actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.AC.critic.parameters(), self.critic_lr)

		self.loss_f = nn.MSELoss()

	def save_models(self, name):
		torch.save(self.AC_T.state_dict(), name+'.pt')

	def load_models(self, name, test=False):

		self.AC_T.load_state_dict(torch.load(name))
		utils.hard_update(self.AC, self.AC_T)
		print("parameters have been loaded")

	def get_exploitation_action(self, state):
		state = Variable(torch.tensor(state)).cuda()
		action = self.actor_T.forward(state).detach().cpu()
		action = action.data.numpy()
		return np.squeeze(action)

	def get_exploration_action(self, state):
		self.steps += 1

		if self.e > self.end_e and self.steps > self.start_training:
			self.e -= (self.initial_e - self.end_e) / 10000

		state = Variable(torch.tensor(state)).cuda()
		action = self.actor.forward(state).detach().cpu()
		action = torch.squeeze(action)
		action = action.data.numpy()

		noise = self.noise.sample()
		action_noise = (1 - self.e) * action + self.e * noise
		action_noise = np.clip(action_noise, self.action_low, self.action_high)
		return action_noise

	def optimize(self):
		s1, a1, r1, s2, done = self.ram.sample(self.batch_size)

		s1 = Variable(torch.tensor(s1)).cuda()
		a1 = Variable(torch.tensor(a1)).cuda()
		r1 = Variable(torch.tensor(r1)).cuda()
		s2 = Variable(torch.tensor(s2)).cuda()

		# optimize critic
		a2 = self.actor_T.forward(s2).detach()
		r_predict = torch.squeeze(self.critic_T.forward(s2, a2).detach())
		r_predict = self.gamma * r_predict
		y_j = r1 + r_predict

		r_ = self.critic.forward(s1, a1)
		r_ = torch.squeeze(r_)

		self.critic_optimizer.zero_grad()

		critic_loss = self.loss_f(y_j, r_)

		critic_loss.backward()
		self.critic_optimizer.step()

		# optimize actor
		pred_a1 = self.actor.forward(s1)
		actor_loss = -1 * torch.mean(self.critic.forward(s1, pred_a1))

		self.actor_optimizer.zero_grad()

		actor_loss.backward()
		self.actor_optimizer.step()

		# update net
		utils.soft_update(self.AC_T, self.AC, self.tau)

		return actor_loss.cpu(), critic_loss.cpu()



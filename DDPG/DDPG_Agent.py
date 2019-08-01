import torch
from torch.nn import functional as F
from torch.autograd import Variable

import gym
import numpy as np
import utils
import DDPG_Model



class DDPGAgent:
	def __init__(self, obs_space, action_space, ram):
		self.obs_dim = obs_space.shape[0]
		self.act_dim = action_space.shape[0] # only for continiuous env

		self.ram = ram
		self.iter = 1
		self.steps = 0
		self.gamma = 0.9
		self.batch_size = 64
		self.initial_e = 0.5
		self.end_e = 0.01
		self.e = self.initial_e

		self.tau = 0.01
		self.critic_lr = 0.001
		self.actor_lr = 0.001
		self.noise = utils.RandomActionNoise(self.act_dim)

		target_net = DDPG_Model.DDPG(self.obs_dim, self.act_dim).cuda()
		learning_net = DDPG_Model.DDPG(self.obs_dim, self.act_dim).cuda()
		utils.hard_update(self.target_net, self.learning_net)

		self.AC = learning_net
		self.AC_T = target_net
		self.actor = self.AC.actor
		self.critic = self.AC.critic
		self.actor_T = self.AC_T.actor
		self.critic_T = self.AC_T.critic

		self.actor_optimizer = torch.optim.Adam(self.AC.actor.parameters(), self.actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.AC.critic.parameters(), self.critic_lr)

	def self.save_models(self, name):
		torch.save(self.AC_T.state_dict(), name+'.pt')

    def load_models(self, name, test=False):
    	# save_state = torch.load(name,
    	# 	map_location=lambda storage, loc:storage)
    	self.AC_T.load_state_dict(torch.load(name))
    	utils.hard_update(self.AC, self.AC_T)
    	# utils.hard_update(self.target_net, self.learning_net)
    	print("parameters have been loaded")

    def get_exploitation_action(self, state):
    	state = Variable(torch.tensor(state)).cuda()
    	action = self.actor_T.forward(state).detach().cpu()
    	return torch.squeeze(action)

    def get_exploration_action(self, state):
    	



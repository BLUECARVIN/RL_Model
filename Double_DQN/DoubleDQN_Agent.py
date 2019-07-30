import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import gym
import numpy as np
import utils
import DQN_Model


class DoubleDQNAgent:
    def __init__(self, obs_space, action_space, ram):
        self.obs_dim = obs_space.shape[0]
        self.act_dim = action_space.n  # only for discrete space
        
        self.ram = ram
        self.iter = 1
        self.steps = 0
        self.gamma = 0.90
        self.batch_size = 64
        self.initial_e = 0.5
        self.end_e = 0.01
        self.e = self.initial_e
        self.target_update_freq = 100
        self.tau = 0.01
        self.lr = 0.001
        
        self.learning_net = DQN_Model.DQN(self.obs_dim, self.act_dim).cuda()
        self.target_net = DQN_Model.DQN(self.obs_dim, self.act_dim).cuda()
        utils.hard_update(self.target_net, self.learning_net)
        
        self.optimizer = torch.optim.Adam(self.learning_net.parameters(), self.lr)
        self.loss_f = nn.MSELoss()
        
    def save_models(self, name):
        torch.save(self.target_net.state_dict(), name+'.pt')

    def load_models(self, name, test=False):
    	# save_state = torch.load(name,
    	# 	map_location=lambda storage, loc:storage)
    	self.target_net.load_state_dict(torch.load(name))
    	utils.hard_update(self.learning_net, self.target_net)
    	# utils.hard_update(self.target_net, self.learning_net)
    	print("parameters have been loaded")

    def get_exploration_action(self, state):
        self.steps += 1
        if np.random.uniform() > self.e:
            state = Variable(torch.tensor(state)).cuda()
            action = torch.argmax(self.learning_net.forward(state)).detach().cpu()
        else:
            action = int(np.random.uniform() * self.act_dim)
        self.e -= (self.initial_e - self.end_e) / 10000
        return action

    def get_exploitation_action(self, state):
    	state = Variable(torch.tensor(state)).cuda()
    	action = torch.argmax(self.target_net.forward(state)).detach().cpu()
    	return action

    def done_state_value(self, r1, y_j, done):
    	for i in range(self.batch_size):
    		if done[i]:
    			y_j[i] = r1[i]
    	return y_j

    def optimize(self):
        s1, a1, r1, s2, done = self.ram.sample(self.batch_size)

        s1 = Variable(torch.tensor(s1)).cuda()
        a1 = Variable(torch.tensor(a1, dtype=torch.int64)).cuda()
        r1 = Variable(torch.tensor(r1)).cuda()
        s2 = Variable(torch.tensor(s2)).cuda()

        self.optimizer.zero_grad()


        # optimize

        #
        action_predict = torch.argmax(self.learning_net.forward(s2), dim=1)
        r_predict = torch.squeeze(self.target_net.forward(s2).gather(1, action_predict.view(-1,1)))
        r_predict = self.gamma * r_predict
        y_j = r1 + r_predict
        y_j = self.done_state_value(r1, y_j, done)

        # r_ : Q(s_j, a_j)
        r_ = self.learning_net.forward(s1)
        r_ = torch.squeeze(r_.gather(1, a1.view(-1,1)))

        # loss: (y_j - Q(s_j, a_j))^2
        loss = self.loss_f(y_j, r_)

        loss.backward()
        self.optimizer.step()

        utils.soft_update(self.target_net, self.learning_net, self.tau)
        self.iter += 1
        return loss.cpu()
import numpy as np
import Memory
import DQN_Agent
import utils

import gym

import torch

def main():
	env = gym.make('CartPole-v0')
	ram = Memory.MemoryBuffer(30000)
	agent = DQN_Agent.DQNAgent(env.observation_space, env.action_space, ram=None)

	steps_done = 0
	episode_reward = []
	count_eps = 0

	agent.load_models('DQN_Test.pt')

	for epoch in range(10000):
		observation = env.reset()
		total_reward = 0
		count_eps += 1
		epoch_loss = []

		env.render()
		print("Epoch:{}".format(epoch))

		for r in range(10000):
			state = np.float32(observation)
			action = agent.get_exploitation_action(state)
			action = np.array(action)

			new_observation, reward, done, _ = env.step(action)

			steps_done += 1
			total_reward += reward
			env.render()

			observation = new_observation
			if done:
				break

		print("\r Total_reward:{}".format(total_reward))

	env.close()

if __name__ == '__main__':
	main()
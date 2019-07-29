import numpy as np
import Memory
import DQN_Agent
import utils

import gym

import torch

def main():
	# env = gym.make('MountainCar-v0')
	env = gym.make('CartPole-v0')
	ram = Memory.MemoryBuffer(500)
	agent = DQN_Agent.DQNAgent(env.observation_space, env.action_space,
		ram)

	steps_done = 0
	episode_reward = []
	max_score = -9999
	count_eps = 0

	for epoch in range(10000):
		observation = env.reset()
		total_reward = 0
		count_eps += 1
		epoch_loss = []
		# print("Epoch:{}".format(epoch))

		for r in range(10000):
			loss = 0
			state = np.float32(observation / 4.8)
			action = agent.get_exploration_action(state)
			action = np.array(action)

			new_observation, reward, done, _ = env.step(action)

			steps_done += 1
			total_reward += reward

			ram.add(observation / 4.8, np.expand_dims(action,axis=0), reward,
				new_observation / 4.8, done)

			env.render()

			observation = new_observation 

			#begin to train
			if steps_done > 100:
				loss = agent.optimize()
				# if steps_done % 300 == 0:
				# 	utils.hard_update(agent.target_net, agent.learning_net)
			epoch_loss.append(loss)
			# print("\r Steps:{}, Total_steps:{}, Actions:{}, Reward:{}, loss:{}".format(
				# r, steps_done, action, reward, loss), end='')
			if done:
				break

		# print("Total_reward:{}, Meanloss:{}, Total_steps:{}".format(total_reward, 
			# torch.mean(torch.tensor(epoch_loss, dtype=torch.float32)),
			# steps_done))
		if epoch % 10 == 0:
			agent.save_models('DQN_Test')
			# print("save model successfully")

		# test the model
		if epoch % 500 == 0:
			test_reward = 0
			for i in range(10):
				observation = env.reset()
				env.render()

				for l in range(10000):
					state = np.float32(observation)
					action = agent.get_exploitation_action(state)
					action = np.array(action)

					new_observation, reward, down, _ = env.step(action)

					test_reward += reward

					env.render()
					observation = new_observation
					if done:
						break
			print("MeanReward:{}".format(test_reward / 10.))


	env.close()

if __name__ == '__main__':
	main()
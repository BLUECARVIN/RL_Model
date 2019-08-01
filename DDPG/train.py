import numpy as np
import Memory
import DDPG_Agent
import utils

import gym

import torch


def main():
	env = gym.make('MountainCarContinuous-v0')
	ram = Memory.MemoryBuffer(1000)
	agent = DDPG_Agent.DDPGAgent(env.observation_space, 
		env.action_space, ram)

	steps_done = 0
	episode_reward = []

	for epoch in range(10000):
		observation = env.reset()
		total_reward = 0
		epoch_actor_loss = []
		epoch_critic_loss = []

		for r in range(10000):
			actor_loss = 0
			critic_loss = 0
			state = np.float32(observation)
			action = agent.get_exploration_action(state)

			new_observation, reward, done, _ = env.step(action)

			steps_done += 1
			total_reward += reward

			# ram.add(observation, np.expand_dims(action, axis=0),
			# 	reward, new_observation, done)
			ram.add(observation, action,
				reward, new_observation, done)

			env.render()

			observation = new_observation

			# begin to train
			if steps_done > 100:
				actor_loss, critic_loss = agent.optimize()

			epoch_actor_loss.append(actor_loss)
			epoch_critic_loss.append(critic_loss)

			if done:
				break

		print("Total_reward:{}, MeanAloss:{}, MeanCloss:{}, Total_steps:{}".format(total_reward, 
			torch.mean(torch.tensor(epoch_actor_loss, dtype=torch.float32)),
			torch.mean(torch.tensor(epoch_critic_loss, dtype=torch.float32)),
			steps_done))
		if epoch % 10 == 0:
			agent.save_models('Pendulum')

	env.close()

if __name__ == '__main__':
	main()
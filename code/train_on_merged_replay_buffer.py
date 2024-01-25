# this is the main file which calls all the remaining functions and
# keeps track of simulation time

import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
#from numba import jit, cuda 
import time
import copy
import numpy as np
import random
import math
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import csv
import pickle
import time
import data_file


from multiprocessing import Pool

def func(_args):
	algo_option = "rl_modified_ddswa"

	train_iter = _args[0]
	sim = _args[1]
	

	if data_file.rl_flag:

		import tensorflow as tf

		if data_file.rl_algo_opt == "DDPG":
			from ddpg_related_class import DDPG as Agent

		elif data_file.rl_algo_opt == "MADDPG":
			from maddpg import DDPG as Agent


		
		ss = [int(5000*len(data_file.arr_rates_to_simulate)), 64]
		actor_lr = 0.0001
		critic_lr = 0.001
		p_factor = 0.0001
		d_factor = 0.99

		max_learn_iter = 100100

		read_file_path = f"../data/merged_replay_buffer_with_next_state/merged_replay_buffer"

		write_trained_policy_file_path = f"../data/merged_replay_buffer_with_next_state/train_sim_{sim}/trained_weights"

		init_weights_path = f"../data/arr_{round(0.1+((sim+1) * 0.01), 2)}/train_homo_stream/train_sim_1/sim_data/trained_weights" # f"../data/merged_replay_buffer_with_next_state/train_sim_{sim}/trained_weights"
		


		#### RL agent object creation ####
		if data_file.rl_algo_opt == "DDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
			
				read_file = open(f"{read_file_path}", 'rb')
				buffer = pickle.load(read_file)
				read_file.close()

				agent.buffer.state_buffer = buffer["state_buffer"]
				agent.buffer.action_buffer = buffer["action_buffer"]
				agent.buffer.reward_buffer = buffer["reward_buffer"]
				agent.buffer.next_state_buffer = buffer["next_state_buffer"]

				agent.buffer.buffer_counter = ss[0]


				for i in range(max_learn_iter):
					agent.buffer.learn()

					agent.update_target(agent.target_actor_.variables, agent.actor_model_.variables, agent.tau_)
					agent.update_target(agent.target_critic_.variables, agent.critic_model_.variables, agent.tau_)
					
					if (i % 5000) == 0:
						agent.actor_model_.save_weights(f"{write_trained_policy_file_path}/actor_weights_itr_{i}")

						agent.actor_model_.save_weights(f"{write_trained_policy_file_path}/actor_weights_itr_final")
						agent.critic_model_.save_weights(f"{write_trained_policy_file_path}/critic_weights_itr_final")

					print(f"learning iteration: {i+1} out of {max_learn_iter}", end="\r")

if __name__ == '__main__':

	args = []
	
	for _train_iter in range(1):
		for _sim_num in range(1, 11):
			args.append([_train_iter, _sim_num])


	pool = Pool(10)

	pool.map(func, args)





import data_file

import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'

import numpy as np

import pickle



arr_rate_array = data_file.arr_rates_to_simulate

write_file_path = f"../data/merged_replay_buffer_with_next_state/"

individual_buffer_size = 5000

merged_buffer_size = individual_buffer_size*len(arr_rate_array)

state_size = data_file.num_features*data_file.num_veh

action_size = data_file.num_veh


for train_sim in range(1, 2):


	merged_replay_buffer = {}

	merged_replay_buffer["state_buffer"] = np.zeros((merged_buffer_size, state_size))

	merged_replay_buffer["action_buffer"] = np.zeros((merged_buffer_size, action_size))

	merged_replay_buffer["reward_buffer"] = np.zeros((merged_buffer_size, 1))

	merged_replay_buffer["next_state_buffer"] = np.zeros((merged_buffer_size, state_size))

	for ind, arr_rate in enumerate(arr_rate_array):

		read_file = open(f"../data/arr_{arr_rate}/train_homo_stream/train_sim_{train_sim}/replay_buffer_sim_{train_sim}", 'rb')

		buffer = pickle.load(read_file)

		read_file.close()

		merged_replay_buffer["state_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["state_buffer"]

		merged_replay_buffer["action_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["action_buffer"]

		merged_replay_buffer["reward_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["reward_buffer"]

		merged_replay_buffer["next_state_buffer"][individual_buffer_size*ind: individual_buffer_size*(ind + 1)] = buffer["next_state_buffer"]

		print(f"train sim: {train_sim}, arrival rate: {arr_rate}")


	dbfile = open(f"{write_file_path}/merged_replay_buffer", 'wb')
	pickle.dump(merged_replay_buffer, dbfile)
	dbfile.close()






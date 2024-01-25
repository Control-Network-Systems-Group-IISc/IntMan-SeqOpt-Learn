# this is the main file which calls all the remaining functions and
# keeps track of simulation time

import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import sys
import shutil
#from numba import jit, cuda 
import time
import copy
import numpy as np
import random
import math
from collections import deque
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import csv
import pickle
import time
import data_file
import vehicle
import gen_vehi
import prov_phase
import coord_phase
import functions
import set_class
import vehicle

from multiprocessing import Pool

def func(_args):

	train_iter = _args[0]
	sim = _args[1]
	arr_rate_array = _args[2]
	arr_rate_ = _args[3]
	train_sim = _args[4]
	### algorithm option if data_file.rl_flag is 0 ###

	algo_option = data_file.algo_option # "comb_opt"

	### algorithm option if data_file.rl_flag is 0 ###

	capture_snapshot_flag = 0

	learning_flag = 0

	max_rep_sim = 1

	time_to_capture = 100

	### flag to run combined optimization for testing RL performance ###

	comb_test_probability = 0

	baseline_test_flag = 0

	baseline_test_freq = 10

	### flag to run combined optimization for testing RL performance ###

	if data_file.rl_flag:

		import tensorflow as tf

		if data_file.rl_algo_opt == "DDPG":
			from ddpg_related_class import DDPG as Agent

		elif data_file.rl_algo_opt == "MADDPG":
			from maddpg import DDPG as Agent


		#### initializing variables to store values ####

		rl_ret_collection = []
		comb_opt_test = []
		moving_avg_ret = []
		ddswa_comp_ret = []
		comb_opt_ret_collection = []
		rl_explore_data = []

		#### initializing variables to store values ####


		
		### algorithm option if data_file.rl_flag is 1 ###

		algo_option = "rl_modified_ddswa"

		### algorithm option if data_file.rl_flag is 1 ###

		if train_iter == -1:
			learning_flag = 1

		else:
			learning_flag = 0


		### flag to switch between stream learning and snapshot learning ####

		stream_rl_flag = 1
		
		### flag to switch between stream learning and snapshot learning ####

		#sim = 1
		ss = [5000, 64]
		actor_lr = 0.0001
		critic_lr = 0.001
		p_factor = 0.0001
		d_factor = 0

		agent = None
		
		#### RL agent object creation ####
		if data_file.rl_algo_opt == "DDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
				

				if learning_flag:
					# agent.actor_model_.load_weights(f"../data/init_weights/train_sim_{sim}/actor_weights_itr_0")
					# agent.critic_model_.load_weights(f"../data/init_weights/train_sim_{sim}/critic_weights_itr_0")

					curr_state = None
					prev_state = None

					curr_act = None
					prev_act = None

					curr_rew = None
					prev_rew = None

					max_rep_sim = 61


			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		
		elif data_file.rl_algo_opt == "MADDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(algo_opt=algo_option, num_of_agents=data_file.max_vehi_per_lane*data_file.lane_max, state_size=data_file.num_features, action_size=2)

			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		
		if (not learning_flag) and (data_file.used_heuristic == None):
			agent.actor_model_.load_weights(f"../data/merged_replay_buffer_with_next_state/train_sim_{train_sim}/trained_weights/actor_weights_itr_{train_iter}")
			max_rep_sim = 1

		#### RL agent object creation ####

	if data_file.run_coord_on_captured_snap:
		with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_info_{sim}.csv", "a", newline="") as f:
			writer = csv.writer(f)
			writer.writerows([["sim_time", "num veh in Vs", "num veh in Vp",  "num veh crossed", "RL like cost", "actual coord cost", "prov phase time", "total coord phase time", "coord phase solve only time per veh", "coord phase object construction plus solve time per veh"]])		

	if data_file.run_coord_on_captured_snap:

		read_captured_file_path = f"../data/captured_snaps/arr_{arr_rate_}/sim_{sim}"

		list_of_snap_dump = []

		done_snaps = 0

		for file_name in os.listdir(f"{read_captured_file_path}"):
			file = open(f"{read_captured_file_path}/{file_name}",'rb')
			object_file = pickle.load(file)
			file.close()
			num_of_veh = functions.get_num_of_objects(object_file[0].prov_veh)
			if num_of_veh > 7:
				continue
			snap_already_done_flag = 0
			try:
				with open(f'../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_info_{sim}.csv', newline='') as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:

						try:
							if round(float(row[0]), 1) == round(float(object_file[1]), 1):
								snap_already_done_flag = 1
								break

						except Exception as e_1:
							...

				if snap_already_done_flag:
					continue
				else:
					list_of_snap_dump.append(object_file)

			except Exception as e:
				list_of_snap_dump.append(object_file)

		for snap_data in list_of_snap_dump:

			print(f"done_snaps: {done_snaps} out of {len(list_of_snap_dump)} in sim {sim}.....", end="\r")			

			sim_obj = snap_data[0]
			time_track = snap_data[1]

			num_of_veh = functions.get_num_of_objects(sim_obj.prov_veh)

			

			prov_veh_copy = copy.deepcopy(sim_obj.prov_veh)
			coord_veh_copy = copy.deepcopy(sim_obj.coord_veh)

			prov_veh_copy__copy = copy.deepcopy(prov_veh_copy)
			coord_veh_copy__copy = copy.deepcopy(coord_veh_copy)

			len_lane_prov_set = [0 for _ in data_file.lanes]
			len_lane_coord_set = [0 for _ in data_file.lanes]

			
			for _l in data_file.lanes:
				len_lane_prov_set[_l] = copy.deepcopy(len(prov_veh_copy__copy[_l]))
				len_lane_coord_set[_l] = copy.deepcopy(len(coord_veh_copy__copy[_l]))

			coord_time_init = time.time()			
			sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para, _, solve_only_time_per_veh, coord_phase_time_per_veh = coord_phase.coord_algo(time_track, prov_veh_copy__copy, coord_veh_copy__copy, algo_option, agent, None, 1, sim, train_iter, sim)
			coord_dura = time.time() - coord_time_init

			with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_info_{sim}.csv", "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows([[time_track, functions.get_num_of_objects(coord_veh_copy), num_of_veh, num_of_veh - functions.get_num_of_objects(prov_veh_copy__copy), coord_cost_with_comb_opt_para/num_of_veh, cp_cost, 0, coord_dura, solve_only_time_per_veh, coord_phase_time_per_veh]])		

			with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_Vc_{sim}.csv", "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows([[time_track, len_lane_prov_set]])

			with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_Vs_{sim}.csv", "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows([[time_track, len_lane_coord_set]])

			done_snaps +=1

			print(f"done_snaps: {done_snaps} out of {len(list_of_snap_dump)} in sim {sim}.....", end="\r")







	else:

		### if random time-varying traff ###

		# list_of_arr_to_sample = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]

		for rep_sim in range(0, max_rep_sim):

			time_track = 0  # time tracking variable

			# ### if random time-varying traff ###

			# for lane in data_file.lanes:
			# 	arr_rate_array[lane] = random.choice(list_of_arr_to_sample)


			# ### if time-varying burst mode traff ###

			# for lane in data_file.lanes:
			# 	arr_rate_array[lane] = 0.15


			cumulative_throuput_count = 0  # cumulative throuput

			throughput_id_in_lane = [0 for _ in data_file.lanes] # variable to help track last vehicle in each lane

			sim_obj = set_class.sets() # a set of sets to classify vehicles whether they are unspawned, in provisional phase, in coordinated phase or have crossed the region of interest

			if not data_file.real_time_spawning_flag:
				file = open(f"../data/compare_files/homogeneous_traffic/arr_{arr_rate_}/sim_obj_num_{sim}", 'rb')
				sim_obj = pickle.load(file)
				file.close()
				total_veh_in_simulation = functions.get_num_of_objects(sim_obj.unspawned_veh)

			if data_file.real_time_spawning_flag:
				veh_id_var = 0
				next_spawn_time = [100 + data_file.max_sim_time for _ in data_file.lanes]
				for lane in data_file.lanes:
					if not (arr_rate_array[lane] == 0):
						next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)


			# if data_file.used_heuristic == "fifo":
			wait_till_time_on_lane = {}
			for lane in data_file.lanes:
				wait_till_time_on_lane[lane] = 0

			while (time_track < data_file.max_sim_time):

				# ### if random time-varying traff ###

				# if (time_track % 100) == 0:
				# 	for lane in data_file.lanes:
				# 		arr_rate_array[lane] = random.choice(list_of_arr_to_sample)


				# ### if time-varying burst mode traff ###

				# if (time_track % 30) == 0:
				# 	for lane in data_file.lanes:
				# 		arr_rate_array[lane] = 0.15

				# if (time_track % 30) == 10:
				# 	for lane in data_file.lanes:
				# 		arr_rate_array[lane] = 0.05

				curr_time = time.time()

				if (not data_file.real_time_spawning_flag):
					if (functions.get_num_of_objects(sim_obj.done_veh) >= total_veh_in_simulation):
						break

				if data_file.rl_algo_opt == "DDPG" and learning_flag and (agent.buffer.buffer_counter > 0) and ((time_track % 1) == 0):
					agent.buffer.learn()
					agent.update_target(agent.target_actor_.variables, agent.actor_model_.variables, agent.tau_)
					agent.update_target(agent.target_critic_.variables, agent.critic_model_.variables, agent.tau_)

				if learning_flag and (time_track % 100) == 0:
					agent.actor_model_.save_weights(f"../data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/sim_data/trained_weights/actor_weights_itr_{int(time_track)}")

				### provisional phase and preparation for coordinated phase ###

				for lane in data_file.lanes:

					if (data_file.real_time_spawning_flag) and (round(time_track, 1) >= round(next_spawn_time[lane], 1)) and (len(sim_obj.unspawned_veh[lane]) == 0) and (not (arr_rate_array[lane] == 0)):
						next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)
						new_veh = vehicle.Vehicle(lane, data_file.int_start[lane], 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array)
						new_veh.id = copy.deepcopy(veh_id_var)
						new_veh.sp_t = copy.deepcopy(time_track)
						veh_id_var += 1
						sim_obj.unspawned_veh[lane].append(copy.deepcopy(new_veh))

					n = len(sim_obj.unspawned_veh[lane])
					v_itr = 0
					while v_itr < n:
						v = sim_obj.unspawned_veh[lane][v_itr]

						pre_v = None
						if len(sim_obj.prov_veh[lane]) > 0:
							pre_v = sim_obj.prov_veh[lane][-1]

						elif len(sim_obj.coord_veh[lane]) > 0:
							pre_v = sim_obj.coord_veh[lane][-1]

						if (round(v.sp_t,1) < round(time_track,1)) and (functions.check_init_config(v, pre_v, time_track)):
							v.sp_t = round(time_track,1)
							prov_sucess = False
							prov_anomaly = -1
							prov_anomaly += 1
							prov_veh = copy.deepcopy(v)
							prov_veh, prov_sucess = prov_phase.prov_phase(prov_veh, pre_v, time_track, wait_till_time_on_lane)

							if data_file.used_heuristic == "fifo":
								sim_obj.coord_veh[lane].append(copy.deepcopy(prov_veh))
							else:
								sim_obj.prov_veh[lane].append(copy.deepcopy(prov_veh))

							sim_obj.unspawned_veh[lane].popleft()
							n = len(sim_obj.unspawned_veh[lane])

							if data_file.used_heuristic == "fifo":
								e_time_p_val = list(filter(lambda posi: posi > (prov_veh.intsize + prov_veh.length), prov_veh.p_traj))[0]
								e_time_index = list(prov_veh.p_traj).index(e_time_p_val) + 1
								prov_veh.exit_time = prov_veh.t_ser[e_time_index]
								functions.storedata(prov_veh, 0, sim, 0)
								wait_till_time_on_lane[prov_veh.lane] = max(wait_till_time_on_lane[prov_veh.lane], prov_veh.exit_time)
							break

						else:
							if data_file.real_time_spawning_flag:
								next_spawn_time[lane] = round(next_spawn_time[lane] + data_file.dt, 1)
							break
				### provisional phase and preparation for coordinated phase done ###

				if data_file.used_heuristic == "fifo":
					print("current time:", time_track, "sim:", sim, "train_sim: ", train_sim, "train_iter:", train_iter, "arr_rate: ", arr_rate_, "heuristic:", data_file.used_heuristic, "................", end="\r")
					time_track = round(time_track + data_file.dt, 1)

					for l in data_file.lanes:
						n_in_coord = len(sim_obj.coord_veh[l])
						v_ind = 0
						while v_ind < n_in_coord:
							coord_while_flag = 0
							v = sim_obj.coord_veh[l][v_ind]
							t_ind = functions.find_index(v, time_track)

							if (t_ind == None) or (v.p_traj[t_ind] > (v.intsize + v.length - v.int_start)):
										
								sim_obj.done_veh[v.lane].append(v)
								sim_obj.coord_veh[v.lane].popleft()
								#del v
								n_in_coord -= 1

							else:
								break

					continue


				### coordinated phase starting ###
				if (round((time_track % data_file.t_opti),1) == 0):

					comb_test_flag = 0

					if np.random.uniform(0,1) < comb_test_probability:
						comb_test_flag = 1

					prov_time = time.time() - curr_time

					if (functions.get_num_of_objects(sim_obj.prov_veh) > 0) and (capture_snapshot_flag == 1):
						m = {}
						m[0] = copy.deepcopy(sim_obj)
						m[1] = time_track
						m[2] = arr_rate_
						dbfile = open(f'../data/captured_snaps/arr_{arr_rate_}/sim_{sim}/time_{time_track}', 'wb')
						pickle.dump(m, dbfile)
						dbfile.close()

					success = False


					if data_file.rl_flag:
						num_of_veh = functions.get_num_of_objects(sim_obj.prov_veh)

						### making copies to run different algorithms

						prov_veh_copy = copy.deepcopy(sim_obj.prov_veh)
						coord_veh_copy = copy.deepcopy(sim_obj.coord_veh)

						prov_veh_copy__copy = copy.deepcopy(prov_veh_copy)
						coord_veh_copy__copy = copy.deepcopy(coord_veh_copy)

						if comb_test_flag:

							prov_set_copy__comb_opt = copy.deepcopy(prov_veh_copy)
							coord_veh_copy__comb_opt = copy.deepcopy(coord_veh_copy)

						### making copies to run different algorithms

						len_lane_prov_set = [0 for _ in data_file.lanes]
						len_lane_coord_set = [0 for _ in data_file.lanes]

						for _l in data_file.lanes:
							len_lane_prov_set[_l] = copy.deepcopy(len(prov_veh_copy__copy[_l]))
							len_lane_coord_set[_l] = copy.deepcopy(len(coord_veh_copy__copy[_l]))

						if learning_flag:
							prev_state = curr_state
							prev_act = curr_act
							prev_rew = curr_rew

						coord_init_time = time.time()
						sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para, state_t, action_t, success = coord_phase.coord_algo(time_track, prov_veh_copy__copy, coord_veh_copy__copy, algo_option, agent, learning_flag, 1, sim, train_iter, train_sim)
						coord_time = time.time() - coord_init_time

						if learning_flag:
							curr_state = state_t
							curr_act = action_t
							curr_rew = 0

						if comb_test_flag:
							comb_test_start = time.time()
							_, comb_cost_test, _, _, _, _, _ = coord_phase.coord_algo(time_track, prov_set_copy__comb_opt, coord_veh_copy__comb_opt, "comb_opt", None, None, 1)
							comb_test_dura = time.time() - comb_test_start

						if (num_of_veh > 0):
							rl_ret = coord_cost_with_comb_opt_para / num_of_veh
							curr_rew = rl_ret
							rl_ret_collection.append(rl_ret)
							if comb_test_flag:
								comb_opt_test.append(comb_cost_test / num_of_veh)
							else:
								comb_opt_test.append(0)

						else:
							rl_ret = None


						if (algo_option == "rl_modified_ddswa") and (learning_flag):

							if (not (prev_rew == None)) and (not (len(prev_state) == 0)):
								agent.buffer.remember((prev_state, prev_act, prev_rew, curr_state))

								qwe = {}
								qwe["state_buffer"] = agent.buffer.state_buffer
								qwe["action_buffer"] = agent.buffer.action_buffer
								qwe["reward_buffer"] = agent.buffer.reward_buffer
								qwe["next_state_buffer"] = agent.buffer.next_state_buffer
								
								dbfile = open(f'../data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/replay_buffer_sim_{sim}', 'wb')
								pickle.dump(qwe, dbfile)
								dbfile.close()

						sim_obj.prov_veh = [deque([]) for _ in range(len(data_file.lanes))]

						if functions.get_num_of_objects(prov_veh_copy__copy) != 0:
							for lane in data_file.lanes:
								for v in prov_veh_copy__copy[lane]:
									pre_v = None
									if len(sim_obj.prov_veh[lane]) > 0:
										pre_v = sim_obj.prov_veh[lane][-1]

									elif len(sim_obj.coord_veh[lane]) > 0:
										pre_v = sim_obj.coord_veh[lane][-1]

									prov_veh = copy.deepcopy(v)
									prov_sucess = False
									prov_anomaly = -1
									prov_anomaly += 1
									prov_veh = copy.deepcopy(v)
									prov_veh, prov_sucess = prov_phase.prov_phase(prov_veh, pre_v, time_track, wait_till_time_on_lane)

									assert len(prov_veh.p_traj) > len(v.p_traj)

									sim_obj.prov_veh[lane].append(copy.deepcopy(prov_veh))


						### RL done ###


						### code for DDSWA or combined optimization ###

					else:
						num_of_veh = functions.get_num_of_objects(sim_obj.prov_veh)
						prov_veh_copy = copy.deepcopy(sim_obj.prov_veh)
						coord_veh_copy = copy.deepcopy(sim_obj.coord_veh)

						prov_veh_copy__copy = copy.deepcopy(prov_veh_copy)
						coord_veh_copy__copy = copy.deepcopy(coord_veh_copy)

						len_lane_prov_set = [0 for _ in data_file.lanes]
						len_lane_coord_set = [0 for _ in data_file.lanes]

						for _l in data_file.lanes:
							len_lane_prov_set[_l] = copy.deepcopy(len(prov_veh_copy__copy[_l]))
							len_lane_coord_set[_l] = copy.deepcopy(len(coord_veh_copy__copy[_l]))

						coord_time_init = time.time()			
						sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para, _, solve_only_time_per_veh, coord_phase_time_per_veh = coord_phase.coord_algo(time_track, prov_veh_copy__copy, coord_veh_copy__copy, algo_option, None, None, 1, sim, train_iter, sim)
						coord_dura = time.time() - coord_time_init
						

						if num_of_veh > 0:
							with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_info_{sim}.csv", "a", newline="") as f:
								writer = csv.writer(f)
								writer.writerows([[time_track, functions.get_num_of_objects(coord_veh_copy), num_of_veh,  num_of_veh - functions.get_num_of_objects(prov_veh_copy__copy), coord_cost_with_comb_opt_para/num_of_veh, cp_cost, prov_time, coord_dura, solve_only_time_per_veh, coord_phase_time_per_veh]])		

							with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_Vc_{sim}.csv", "a", newline="") as f:
								writer = csv.writer(f)
								writer.writerows([[time_track, len_lane_prov_set]])

							with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_Vs_{sim}.csv", "a", newline="") as f:
								writer = csv.writer(f)
								writer.writerows([[time_track, len_lane_coord_set]])

						sim_obj.prov_veh = [deque([]) for _ in range(len(data_file.lanes))]

						if functions.get_num_of_objects(prov_veh_copy__copy) != 0:
							for lane in data_file.lanes:
								for v in prov_veh_copy__copy[lane]:
									pre_v = None
									if len(sim_obj.prov_veh[lane]) > 0:
										pre_v = sim_obj.prov_veh[lane][-1]

									elif len(sim_obj.coord_veh[lane]) > 0:
										pre_v = sim_obj.coord_veh[lane][-1]

									prov_veh = copy.deepcopy(v)
									prov_sucess = False
									prov_anomaly = -1
									prov_anomaly += 1
									prov_veh = copy.deepcopy(v)
									prov_veh, prov_sucess = prov_phase.prov_phase(prov_veh, pre_v, time_track, wait_till_time_on_lane)

									assert len(prov_veh.p_traj) > len(v.p_traj)

									sim_obj.prov_veh[lane].append(copy.deepcopy(prov_veh))

						### DDSWA or combined optimization done ###
				
				### coordinated phase done ###


				### update current time###
				time_track = round((time_track + data_file.dt), 1)
				if learning_flag:
					...
					print(f"arr_rate: {arr_rate_}, rep: {rep_sim}", "current time:", time_track, "sim:", sim, "train_iter:", train_iter, end="\r")
				else:
					print("current time:", time_track, "sim:", sim, "train_sim: ", train_sim, "train_iter:", train_iter, "arr_rate: ", arr_rate_, "heuristic:", data_file.used_heuristic, "................", end="\r")

				### update current time###


				### throuput calculation ###
				for l in data_file.lanes:
					for v in sim_obj.coord_veh[l]:
						t_ind = functions.find_index(v, time_track)

						if  ((t_ind == None) or (v.p_traj[t_ind] > (v.intsize + data_file.L))) and (v.id >= throughput_id_in_lane[l]):

							throughput_id_in_lane[l] = copy.deepcopy(v.id)
							cumulative_throuput_count += 1
						
						else:
							break

				### throuput calculation ###




				### removing vehicles which have crossed the region of interest ###
				for l in data_file.lanes:
					n_in_coord = len(sim_obj.coord_veh[l])
					v_ind = 0
					while v_ind < n_in_coord:
						coord_while_flag = 0
						v = sim_obj.coord_veh[l][v_ind]
						t_ind = functions.find_index(v, time_track)

						if (t_ind == None) or (v.p_traj[t_ind] > (v.intsize + v.length - v.int_start)):
									
							sim_obj.done_veh[v.lane].append(v)
							sim_obj.coord_veh[v.lane].popleft()
							n_in_coord -= 1

						else:
							break

				### removed vehicles which have crossed the region of interest ###

			### plotting and other processing
			if data_file.rl_flag:
				wri = []
				for k in range(len(rl_ret_collection)):
					if comb_test_flag:
						_wri = [rl_ret_collection[k], comb_opt_test[k], comb_opt_test[k]-rl_ret_collection[k]]  # , (a_t[0]), (a_t[2]), (a_t[4]), (a_t[6]), (a_t[8]), (a_t[10]),
						wri.append(_wri)
						
				plt.clf()
				plt.plot(range(len(rl_ret_collection)), rl_ret_collection, "o")
				plt.plot(range(len(comb_opt_test)), comb_opt_test, "v")
				if learning_flag or (data_file.used_heuristic != None):
					...
				else:
					plt.savefig(f"../data/arr_{arr_rate_}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/sim_{sim}_train_iter_{train_iter}.png", dpi=300)

	if not (data_file.used_heuristic == None):
		with open(f"../data/{data_file.used_heuristic}/arr_{arr_rate_}/done_log.csv", "a", newline="") as f:
			writer = csv.writer(f)
			writer.writerows([[sim]])		




if __name__ == '__main__':

	arr_rates_to_sim = data_file.arr_rates_to_simulate

	args = []


	if data_file.used_heuristic == None:
	
		if data_file.rl_flag:

			train_or_test = str(sys.argv[1])

			if train_or_test == "--train":
				for _train_iter in range(1):
					for _sim_num in range(1, 2):
						for _arr_rate_ in arr_rates_to_sim:
							arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_}
							args.append([-1, _sim_num, arr_rate_array_, _arr_rate_, 0])
							# func(args[-1])
				pool = Pool(10)
				pool.map(func, args)

			elif train_or_test == "--test":


				if not data_file.run_coord_on_captured_snap:

					_train_iter_list = [int(sys.argv[2])]

					for _train_iter in _train_iter_list:
						for _sim_num in range(1, 11):
							for _train_sim in list(range(2, 11)):
								for _arr_rate_ in arr_rates_to_sim:
									arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_}
									save_path_pickobj = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}"

									file_path = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}/sim_{_sim_num}/sim_{_sim_num}_train_iter_{_train_iter}.png"

									try:
										with open(f"{file_path}") as f:
											f.close()

									except:
										shutil.rmtree(f'{save_path_pickobj}/pickobj_sim_{_sim_num}')
										os.mkdir(f'{save_path_pickobj}/pickobj_sim_{_sim_num}')
										args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])
										print(f"train_sim: {_train_sim}, train_iter: {_train_iter}, sim: {_sim_num}")

					pool = Pool(10)
					pool.map(func, args)

				else:
					_arr_rate_ = 0.08
					for _sim_num in range(1,4):
						args.append([5000,_sim_num,0,_arr_rate_,8])
						func(args[-1])				




		elif not data_file.run_coord_on_captured_snap:
			for _arr_rate_ in arr_rates_to_sim:
				arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_}
				for _sim_num in range(1, 101):
					args.append([0, _sim_num, arr_rate_array_, _arr_rate_, 0])
					func(args[-1])


		else:
			_arr_rate_ = 0.08
			for _sim_num in range(1,4):
				args.append([0,_sim_num,0,_arr_rate_,0])
				func(args[-1])


	else:

		for _train_iter in [0]:
			for _sim_num in range(1, 101):
				for _train_sim in list(range(1)):
					for _arr_rate_ in arr_rates_to_sim:

						#### regular ####
						arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_}
						
						### hetero-traff ###
						# arr_rate_array_ = {0:0, 1:0.13, 2:0.18, 3:0, 4:0.08, 5:0.15, 6:0, 7:0.19, 8:0.09, 9:0, 10:0.05, 11:0.16}

						heuristics_save_path = f"../data/{data_file.used_heuristic}/arr_{_arr_rate_}"

						try:

							flag_temp_ = True
							with open(f'{heuristics_save_path}/done_log.csv', newline='') as csvfile:
								reader = csv.reader(csvfile)
								for row in reader:
									if int(row[0]) == _sim_num:
										flag_temp_ = False
										break

							if flag_temp_:			
								print(f"arr_rate: {_arr_rate_}\tsim: {_sim_num}")
								shutil.rmtree(f'{heuristics_save_path}/pickobj_sim_{_sim_num}')
								os.mkdir(f'{heuristics_save_path}/pickobj_sim_{_sim_num}')
								args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])

						except:
							print(f"arr_rate: {_arr_rate_}\tsim: {_sim_num}")
							shutil.rmtree(f'{heuristics_save_path}/pickobj_sim_{_sim_num}')
							os.mkdir(f'{heuristics_save_path}/pickobj_sim_{_sim_num}')
							args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])

		pool = Pool(18)
		pool.map(func, args)









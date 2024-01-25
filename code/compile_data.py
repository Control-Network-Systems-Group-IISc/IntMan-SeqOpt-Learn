import os
import numpy as np
import csv
from matplotlib import pyplot as plt
import pickle

import data_file



def compile_func(_tr_iter_):

	avg_time_to_cross_norm_dist_and_vel_all_arr = []

	avg_time_to_cross_comb_opt_all_arr = []

	avg_obj_fun_norm_dist_and_vel_all_arr = []

	avg_obj_fun_comb_opt_all_arr = []

	sp_t_limit = 300


	avg_true_arr_rate = []

	heuristic = data_file.used_heuristic

	write_path = f"../data/{heuristic}"

	if heuristic == None:
		write_path = f"../data/"

	arr_rate_times_100_array = data_file.arr_rates_to_simulate #list(range(1, 11)) #+ list(range(20, 100, 10))

	for arr_rate in arr_rate_times_100_array:

		train_iter_list = [_tr_iter_]

		num_train_iter = len(train_iter_list)

		if heuristic is not None:
			sim_list = list(range(1, 101))
			train_sim_list = list(range(1, 11))

		else:
			sim_list = list(range(1, 11))
			train_sim_list = list(range(1, 11))


		num_sim = len(sim_list)

		test_data = {}
		throughput = {}
		ttc = {}
		all_ttc = []
		exit_vel = {}

		percentage_comparison_dict = {}
		throughput_ratio_dict = {}


		total_comb_opt_veh = {}

		comb_opt_veh_dict = {}

		
		total_veh_num = {}
		heuristic_veh_dict = {}
		for train_iter in train_iter_list:
			test_data[train_iter] = {}
			throughput[train_iter] = {}
			ttc[train_iter] = {}
			exit_vel[train_iter] = {}
			total_veh_num[train_iter] = {}
			for train_sim in train_sim_list:
				test_data[train_iter][train_sim] = {}
				throughput[train_iter][train_sim] = {}
				ttc[train_iter][train_sim] = {}
				exit_vel[train_iter][train_sim] = {}
				total_veh_num[train_iter][train_sim] = {}
				for sim in sim_list:

					if heuristic == None:
						test_data_file_path = f"../data/arr_{arr_rate}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}"

					else:
						test_data_file_path = f"../data/{heuristic}/arr_{arr_rate}"
					
					test_data[train_iter][train_sim][sim] = 0
					throughput[train_iter][train_sim][sim] = 0
					ttc[train_iter][train_sim][sim] = 0
					exit_vel[train_iter][train_sim][sim] = 0
					total_veh_num[train_iter][train_sim][sim] = 0

					veh_num = 0
					temp = 0
					temp_ttc = 0
					temp_exit_vel = 0

					for c in os.listdir(f"{test_data_file_path}/pickobj_sim_{sim}"):
						try:
							file = open(f"{test_data_file_path}/pickobj_sim_{sim}/{c}",'rb')
							object_file = pickle.load(file)
							file.close()
						except:
							continue

						if (object_file[int(c)].sp_t > sp_t_limit) or (object_file[int(c)].sp_t < 90.5):
							continue
						
						else:
							veh_num += 1
							try:  
								temp += object_file[int(c)].priority * (object_file[int(c)].p_traj[int(data_file.T_sc/data_file.dt) -1] - object_file[int(c)].p0)
								index_var = 0
								for time, pos in zip(object_file[int(c)].t_ser, object_file[int(c)].p_traj):
									if pos >= object_file[int(c)].length + object_file[int(c)].intsize:
										temp_ttc += (time - object_file[int(c)].t_ser[0]) * object_file[int(c)].priority
										all_ttc.append(time - object_file[int(c)].t_ser[0])
										throughput[train_iter][train_sim][sim] += 1
										temp_exit_vel += object_file[int(c)].v_traj[index_var]
										break

									index_var += 1
							
							except IndexError:
								continue
								
					total_veh_num[train_iter][train_sim][sim] = veh_num
					print(f"heuristic: {heuristic}, arr_rate: {arr_rate}, sim: {sim}, veh_num: {veh_num}, ...................", end="\r") # 
					test_data[train_iter][train_sim][sim] += temp

					ttc[train_iter][train_sim][sim] += temp_ttc/total_veh_num[train_iter][train_sim][sim]
					exit_vel[train_iter][train_sim][sim] += temp_exit_vel/total_veh_num[train_iter][train_sim][sim]

		temp_obj_list = []
		temp_arr_list = []
		for train_iter in train_iter_list: 
			for train_sim in train_sim_list:
				for sim in sim_list:
					temp_obj_list.append(test_data[train_iter][train_sim][sim])
					temp_arr_list.append(total_veh_num[train_iter][train_sim][sim])

		avg_obj_fun_norm_dist_and_vel_all_arr.append( np.average(np.asarray(temp_obj_list)) )
		avg_true_arr_rate.append(np.average(np.asarray(temp_arr_list)))



		average_percentage_comparison_list = [0 for _ in range(num_train_iter)]

		var_percentage_comparison_list = [0 for _ in range(num_train_iter)]

		average_throughput_ratio_list = [0 for _ in range(num_train_iter)]

		var_throughput_ratio_list = [0 for _ in range(num_train_iter)]

		average_exit_vel_list = [0 for _ in range(num_train_iter)]

		var_exit_vel_list = [0 for _ in range(num_train_iter)]

		average_ttc_list = [0 for _ in range(num_train_iter)]

		var_ttc_list = [0 for _ in range(num_train_iter)]


		for train_iter_ind, train_iter in enumerate(train_iter_list):
			temp_var_list = []
			temp_throughput_ratio = []
			temp_ttc_list = []
			temp_exit_vel_list = []

			for train_sim in train_sim_list:
				for sim in sim_list:
					temp_ttc_list.append(ttc[train_iter][train_sim][sim])
					average_ttc_list[train_iter_ind] += ttc[train_iter][train_sim][sim]
					
			var_ttc_list[train_iter_ind] = np.var(np.asarray(temp_ttc_list))
			average_ttc_list[train_iter_ind] = average_ttc_list[train_iter_ind]/(num_sim*len(train_sim_list))

		avg_time_to_cross_norm_dist_and_vel_all_arr.append(average_ttc_list)


		print(f"90%le ttc fir arr {arr_rate} is {np.percentile(np.asarray(all_ttc), 90)}")


	print(f"comb_opt_avg_obj_data: {avg_obj_fun_comb_opt_all_arr}")
	print(f"{heuristic}_avg_obj_data: {avg_obj_fun_norm_dist_and_vel_all_arr}")
	print(f"{heuristic}_avg_true_arr_rate: {avg_true_arr_rate}")
	print(f"average % diff: {average_percentage_comparison_list[train_iter_ind]}")

	with open(f"../data/comb_avg_obj_fun_all_arr.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows([[_] for _ in avg_obj_fun_comb_opt_all_arr])

	with open(f"{write_path}rl_avg_obj_fun_all_arr_{_tr_iter_}.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows([[_] for _ in avg_obj_fun_norm_dist_and_vel_all_arr])


	with open(f"{write_path}rl_avg_true_arr_rate_all_arr_{_tr_iter_}.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows([[_] for _ in avg_true_arr_rate])


	print(f"comb_opt_avg_ttc_data: {avg_time_to_cross_comb_opt_all_arr}")
	print(f"{heuristic}_avg_ttc_data: {avg_time_to_cross_norm_dist_and_vel_all_arr}")

	with open(f"../data/comb_avg_ttc_all_arr.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows([[_] for _ in avg_time_to_cross_comb_opt_all_arr])

	with open(f"{write_path}rl_avg_ttc_all_arr_{_tr_iter_}.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows([[_[0]] for _ in avg_time_to_cross_norm_dist_and_vel_all_arr])



if __name__ == "__main__":

	args_in = []
	iter_list = [100000] #[2000, 5000, 12000] # [7000, 10000, 13000, 14000, 25000, 50000] #[0, 100, 500, 1000, 2000, 3000, 4000, 5000]
	for tr_iter in iter_list:
		args_in.append(tr_iter)
		compile_func(tr_iter)

	
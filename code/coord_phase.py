# this file contains the function coord_phase().
# it takes in a vehicle object, previous vehicle on that lane, current time
# and webster estimate of simulation time as arguments.
# it returns the vehicle object with updated trajectory of the vehicle 
# till the end of its provisional phase
# this file contains the code for provisional phase

import matplotlib.pyplot as plt
import copy
import csv
import coord_opti_class
import data_file
import functions
import prece_dem
import get_states
import pickle
import multiprocessing as mp
import vehicle
#from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool
import time
import casadi
from toolz.itertoolz import partition_all
import os, shutil
import math

from dask.distributed import Client
import dask.multiprocessing
import dask

from dask.distributed import Client
from dask.distributed import as_completed
from dask.distributed import wait
import dask.multiprocessing
import dask


class temp_comb_obj():
	def __init__(self, max_iter_ipopt, coun, step, tot_cost, pos, vel, acc, max_iter_flag, each_v_c, t_eval, comp_time, solve_only_comp_time):
		self.next_max_iter = max_iter_ipopt
		self.count = coun
		self.steps = step
		self.tot_cost = tot_cost
		self.pos = pos
		self.vel = vel
		self.acc_u = acc
		self.max_iter_exceeded = max_iter_flag
		self.each_veh_cost = each_v_c
		self.traf_eval = t_eval
		self.computation_time = comp_time
		self.solve_only_computation_time = solve_only_comp_time

class PickalableSWIG:
	def __setstate__(self, state):
		self.__init__(*state['args'])
	def __getstate__(self):
		return {'args': self.args}

class PickalableCombOpt(coord_opti_class.coord_opti_class_combined_opt, PickalableSWIG):
	def __init__(self, *args):
		self.args = args
		coord_opti_class.coord_opti_class_combined_opt.__init__(self, *args)

class PickalableBestSeq(coord_opti_class.coord_opti_class_seq_opt, PickalableSWIG):
	def __init__(self, *args):
		self.args = args
		coord_opti_class.coord_opti_class_seq_opt.__init__(self, *args)

#@ray.remote
def redef_coord_opti_comb_opt(zip_obj):
	iter_m, count, sc, pr, co, ti, ic =[*(zip_obj)]
	if data_file.algo_option == "comb_opt":
		solute = PickalableCombOpt(iter_m, count, sc, pr, co, ti, ic)
	elif data_file.algo_option == "best_seq":
		solute = PickalableBestSeq(iter_m, count, sc, pr, co, ti, ic)
	temp_obj = temp_comb_obj(solute.next_max_iter, count, solute.steps, solute.tot_cost, solute.pos, solute.vel, solute.acc_u, solute.max_iter_exceeded, solute.each_veh_cost, solute.traf_eval, solute.computation_time, solute.solve_only_computation_time)
	return temp_obj

def get_chunk_res(chunk):
	return [redef_coord_opti_comb_opt(x) for x in chunk]

	

def coord_algo(t_init, prov_set, coord_set, algo_option, rl_agent, learn_flag, exit_time_flag, sim_num, train_iter_num, tr_sim_num, z=-1):
	total_cost = 0
	comb_opt_like_cost = 0
	redo_flag = 0
	explore_flag = 0


	coord_set_copy = copy.deepcopy(coord_set)
	prov_set_copy = copy.deepcopy(prov_set)

	num_prov_set = functions.get_num_of_objects(prov_set)
	num_coord_set = functions.get_num_of_objects(coord_set)

	num_of_vehs_entering_coord_phase = functions.get_num_of_objects(prov_set)
	
	if (algo_option == "comb_opt") or (algo_option == "best_seq"):

		cost_for_all_sequences = []

		init_conds = [[] for _ in data_file.lanes]

		if (functions.get_num_of_objects(prov_set) == 0):
			return coord_set, total_cost, comb_opt_like_cost, [], [], True

		feas_schedules = functions.get_feasible_schedules(prov_set)

		for li in range(len(data_file.lanes)):
			ln = data_file.lanes[li]

			temp_pos_assert = []

			for veh in prov_set[ln]:
				if len(veh.t_ser) > 1:
					temp_ind = functions.find_index(veh, t_init)
					if temp_ind == None:
						print(f"time: {t_init}, t_ser: {veh.t_ser}")
					init_pos0 = veh.p_traj[temp_ind]
					init_vel0 = veh.v_traj[temp_ind]

					veh.coord_init_pos = copy.deepcopy(veh.p_traj[temp_ind])
					veh.coord_init_vel = copy.deepcopy(veh.v_traj[temp_ind])

					veh.p_traj = veh.p_traj[:temp_ind]
					veh.v_traj = veh.v_traj[:temp_ind]
					veh.u_traj = veh.u_traj[:temp_ind]
					veh.t_ser = veh.t_ser[:temp_ind]

				else:
					init_pos0 = veh.p0
					init_vel0 = veh.v0

					veh.coord_init_pos = copy.deepcopy(veh.p0)
					veh.coord_init_vel = copy.deepcopy(veh.v0)

					veh.p_traj = []
					veh.v_traj = []
					veh.u_traj = []
					veh.t_ser = []


				temp_pos_assert.append(init_pos0)

				if len(temp_pos_assert) > 1:
					for h in range(1, len(temp_pos_assert) - 1):
						for o in range(h):
							assert not (round(temp_pos_assert[h], 4) == round(temp_pos_assert[o], 4))

				init_conds[veh.lane].append(copy.deepcopy([init_pos0, init_vel0]))

		assert not (functions.get_num_of_objects(prov_set) == 0)

		max_cost = None
		optimal_sol = None
		temp_ind1 = 1

		results = []
		if len(feas_schedules) > 0:
			dask.config.set({'scheduler.work-stealing': True})
			client = Client("10.64.39.40:8786")

			client.upload_file('data_file.py')
			client.upload_file('functions.py')
			client.upload_file('vehicle.py')
			client.upload_file('coord_opti_class.py')
			client.upload_file('get_states.py')
			client.upload_file('prece_dem.py')
			client.upload_file('coord_phase.py')
			
			default_iter = 1000

			list_of_args = [[default_iter, arg_c, arg_s] for arg_c, arg_s in enumerate(feas_schedules)]

			for itr_c in range(len(feas_schedules)):
				list_of_args[itr_c].append(prov_set)
				list_of_args[itr_c].append(coord_set)
				list_of_args[itr_c].append(t_init)
				list_of_args[itr_c].append(init_conds)

			# list_of_args_ = []
			chunk_size = 50
			for chunk_iter in range(0, int(len(list_of_args)/chunk_size) + 1):
				chunk = list_of_args[chunk_size*chunk_iter: min(chunk_size*(chunk_iter + 1), len(list_of_args))]
				results += client.map(redef_coord_opti_comb_opt, chunk)

			if len(results) != len(feas_schedules):
				print(a)

			any_bad_seq = False

			num_bad_seq = 0

			if any_bad_seq == True:

				all_lane_seq = []
				all_veh_id_seq = []

				for feas_sched in feas_schedules:
					all_lane_seq.append([vehic.lane for vehic in feas_sched])
					all_veh_id_seq.append([vehic.id for vehic in feas_sched])

				with open(f"./data/all_seq_lane.csv", "a", newline="") as f:
					writer = csv.writer(f)
					for lane_seq in all_lane_seq:
						writer.writerows([lane_seq])
						writer.writerows([])

				with open(f"./data/all_seq_veh_id.csv", "a", newline="") as f:
					writer = csv.writer(f)
					for veh_id_seq in all_veh_id_seq:
						writer.writerows([veh_id_seq])
						writer.writerows([])

			computation_time = 0
			solve_only_comp_time = 0
			# print(f"\n*****")
			for itr_ele, elem in enumerate(results):

				element = elem.result() #ray.get(elem)

				if element.max_iter_exceeded == False:

					computation_time += element.computation_time

					solve_only_comp_time += element.solve_only_computation_time

					if max_cost is None:
						max_cost = copy.deepcopy(element.tot_cost)
						optimal_sol = copy.deepcopy(element)
					elif element.tot_cost > max_cost:
						max_cost = element.tot_cost
						optimal_sol = copy.deepcopy(element)

					cost_for_all_sequences.append(element.tot_cost)

				elif not any_bad_seq:
					any_bad_seq = True


			total_cost = max_cost

			if optimal_sol.max_iter_exceeded == True:
				pass

			seq_cost_diff = []
			for seq_cost in cost_for_all_sequences:
				seq_cost_diff.append(max(cost_for_all_sequences) - seq_cost)
			
			##### update exittime #####

			_V_P_set = copy.deepcopy(prov_set)

			flag_some_did_not_cross_in_incomp_lane = [False for _ in data_file.lanes]

			for li in range(len(data_file.lanes)):
				ln = data_file.lanes[li]
				veh_ind = 0

				for veh_ind in range(len(_V_P_set[ln])):

					veh = _V_P_set[ln][veh_ind]

					if ((optimal_sol.pos[ln][veh_ind][-1] > (veh.intsize + veh.length)) and (flag_some_did_not_cross_in_incomp_lane[ln] == False)):
						e_time_p_val = list(filter(lambda posi: posi > (veh.intsize + veh.length), optimal_sol.pos[ln][veh_ind]))[0]
						e_time_index = list(optimal_sol.pos[ln][veh_ind]).index(e_time_p_val)
						
						if len(veh.t_ser) > 0:
							veh.exittime = round((veh.t_ser[-1] + (e_time_index*data_file.dt)), 1)

						else:
							veh.exittime = round((veh.sp_t + (e_time_index*data_file.dt)), 1)

						veh.stime = round(t_init, 1)

						for n in range(optimal_sol.steps):
							veh.p_traj.append(optimal_sol.pos[ln][veh_ind][n])
							veh.v_traj.append(optimal_sol.vel[ln][veh_ind][n])
							veh.u_traj.append(optimal_sol.acc_u[ln][veh_ind][n])
							veh.t_ser.append(round(t_init + (n * data_file.dt), 1))

						for t in range(len(veh.t_ser)):
							veh.finptraj[veh.t_ser[t]] = veh.p_traj[t]
							veh.finvtraj[veh.t_ser[t]] = veh.v_traj[t]
							veh.finutraj[veh.t_ser[t]] = veh.u_traj[t]

						for ti in range(1, len(veh.t_ser)):
							if round((veh.t_ser[ti] - veh.t_ser[ti-1]), 1) > data_file.dt:
								print(veh.lane)
								print(veh.id)
								print(veh.t_ser)

								print(a)

						time_index = functions.find_index(veh, t_init)

						if time_index == None:
							print(a)

						if not round(veh.p_traj[time_index], 4) == round(init_conds[veh.lane][veh_ind][0], 4):
							print(a)

						try:
							veh_stime_index = functions.find_index(veh, t_init)
							veh.comb_opt_like_cost += data_file.W_pos*(veh.p_traj[-1] - veh.p_traj[veh_stime_index])
							veh.comb_opt_like_cost -= sum([data_file.W_acc* veh.u_traj[inde]**2 for inde in range(veh_stime_index, len(veh.u_traj))])
							veh.comb_opt_like_cost -= sum([data_file.W_jrk* (veh.u_traj[inde+1] - veh.u_traj[inde])**2 for inde in range(veh_stime_index, len(veh.u_traj)-1)])

						except Exception as e:
							print("error in computing comb_opt_like_cost", e)

						veh.traffic_eval_func_val = optimal_sol.traf_eval[ln][veh_ind]
						comb_opt_like_cost += veh.comb_opt_like_cost#optimal_sol.each_veh_cost[ln][veh_ind]

						if not data_file.rl_flag:
							functions.storedata(veh, tr_sim_num, sim_num, 0)
						
						coord_set[ln].append(copy.deepcopy(veh))
						rem_ve = prov_set[ln].popleft()
					
					else:
						flag_some_did_not_cross_in_incomp_lane[li] = True
						for lin in veh.incomp:
							flag_some_did_not_cross_in_incomp_lane[lin] = True

				for vehic in prov_set[ln]:
					vehic.p_traj.append(vehic.coord_init_pos)
					vehic.v_traj.append(vehic.coord_init_vel)
					vehic.t_ser.append(t_init)
					comb_opt_like_cost -= data_file.W_pos * (vehic.coord_init_pos - vehic.int_start) ** 2

			client.close()			

		return coord_set, total_cost, comb_opt_like_cost, any_bad_seq, (solve_only_comp_time/num_of_vehs_entering_coord_phase), (computation_time/num_of_vehs_entering_coord_phase)






#------------------------------------------------------------------------------------------------#

	elif (algo_option == "rl_modified_ddswa"):

		init_conds = [[] for _ in data_file.lanes]

		if data_file.used_heuristic == None:
			algo_to_use = "rl_modified_ddswa"

		else:
			algo_to_use = data_file.used_heuristic


		for li in range(len(data_file.lanes)):
			ln = data_file.lanes[li]

			temp_pos_assert = []

			for veh in prov_set[ln]:
				if len(veh.t_ser) > 1:
					temp_ind = functions.find_index(veh, t_init)
					init_pos0 = veh.p_traj[temp_ind]
					init_vel0 = veh.v_traj[temp_ind]

					veh.coord_init_pos = copy.deepcopy(veh.p_traj[temp_ind])
					veh.coord_init_vel = copy.deepcopy(veh.v_traj[temp_ind])

					veh.p_traj = veh.p_traj[:temp_ind]
					veh.v_traj = veh.v_traj[:temp_ind]
					veh.u_traj = veh.u_traj[:temp_ind]
					veh.t_ser = veh.t_ser[:temp_ind]

				else:
					init_pos0 = veh.p0
					init_vel0 = veh.v0

					veh.coord_init_pos = copy.deepcopy(veh.p0)
					veh.coord_init_vel = copy.deepcopy(veh.v0)

					veh.p_traj = []
					veh.v_traj = []
					veh.u_traj = []
					veh.t_ser = []


				temp_pos_assert.append(init_pos0)

				if len(temp_pos_assert) > 1:
					for h in range(1, len(temp_pos_assert) - 1):
						for o in range(h):
							assert not (round(temp_pos_assert[h], 4) == round(temp_pos_assert[o], 4))

				init_conds[veh.lane].append(copy.deepcopy([init_pos0, init_vel0]))

		prov_set, pre_s, dem_s, state, action, explore_flag = prece_dem.get_prece_dem_RL(t_init, prov_set,
																						 coord_set, rl_agent,
																						 algo_to_use,
																						 learn_flag)

		max_cost = None
		optimal_sol = None

		copy_prov_set = copy.deepcopy(prov_set)
		sched_new_algo = []

		# if learn_flag:
		for veh_num__ in range(functions.get_num_of_objects(prov_set)):
			list_of_first = []
			set_f_copy_prov = functions.get_set_f(copy_prov_set)
			for lane in set_f_copy_prov:
				for ve in lane:
					list_of_first.append(ve)

			sched_temp = sorted(list_of_first, key=lambda x: x.priority_index, reverse=True)

			sched_new_algo.append(sched_temp[0])
			copy_prov_set[sched_new_algo[-1].lane].popleft()

		result = coord_opti_class.coord_opti_class_seq_opt(3000, 0, sched_new_algo, prov_set, coord_set, t_init, init_conds)

		max_cost = copy.deepcopy(result.tot_cost)
		optimal_sol = copy.deepcopy(result)

		##### update exittime #####

		_V_P_set = copy.deepcopy(prov_set)

		flag_some_did_not_cross_in_incomp_lane = [False for _ in data_file.lanes]

		for li in range(len(data_file.lanes)):
			ln = data_file.lanes[li]
			veh_ind = 0

			for veh_ind in range(len(optimal_sol.pos[ln])):#_V_P_set[ln])):

				veh = _V_P_set[ln][veh_ind]

				if (optimal_sol.pos[ln][veh_ind][-1] > (veh.intsize + veh.length)) and (flag_some_did_not_cross_in_incomp_lane[ln] == False):

					e_time_p_val = list(filter(lambda posi: posi > (veh.intsize + veh.length), optimal_sol.pos[ln][veh_ind]))[0]
					e_time_index = list(optimal_sol.pos[ln][veh_ind]).index(e_time_p_val) + 1
					
					if len(veh.t_ser) > 0:
						veh.exittime = round((veh.t_ser[-1] + (e_time_index*data_file.dt)), 1)

					else:
						veh.exittime = round((veh.sp_t + (e_time_index*data_file.dt)), 1)

					veh.stime = round(t_init, 1)

					for n in range(optimal_sol.steps):
						veh.p_traj.append(optimal_sol.pos[ln][veh_ind][n])
						veh.v_traj.append(optimal_sol.vel[ln][veh_ind][n])
						veh.u_traj.append(optimal_sol.acc_u[ln][veh_ind][n])
						veh.t_ser.append(round(t_init + (n * data_file.dt), 1))

					for t in range(len(veh.t_ser)):
						veh.finptraj[veh.t_ser[t]] = veh.p_traj[t]
						veh.finvtraj[veh.t_ser[t]] = veh.v_traj[t]
						veh.finutraj[veh.t_ser[t]] = veh.u_traj[t]

					for ti in range(1, len(veh.t_ser)):
						if round((veh.t_ser[ti] - veh.t_ser[ti-1]), 1) > data_file.dt:
							print(veh.lane)
							print(veh.id)
							print(veh.t_ser)
							print(a)

					time_index = functions.find_index(veh, t_init)

					if time_index == None:
						print(a)

					if not round(veh.p_traj[time_index], 4) == round(init_conds[veh.lane][veh_ind][0], 4):
						print(a)


					veh_stime_index = functions.find_index(veh, t_init)

					dist_to_cross = (veh.intsize + veh.length) - veh.p_traj[veh_stime_index]

					dist_to_hit_vm = ((veh.v_max**2) - (veh.v_traj[veh_stime_index]**2)) / (2 * veh.u_max)

					if dist_to_cross >= dist_to_hit_vm:
						tau_hat = ((dist_to_cross - dist_to_hit_vm) / (veh.v_max)) + ((veh.v_max - veh.v_traj[veh_stime_index])/ (veh.u_max))

					else:
						root_p = (-veh.v_traj[veh_stime_index] + np.sqrt((veh.v_traj[veh_stime_index]**2) + (2*veh.u_max*dist_to_cross)) )/ veh.u_max
						root_m = (-veh.v_traj[veh_stime_index] - np.sqrt((veh.v_traj[veh_stime_index]**2) + (2*veh.u_max*dist_to_cross)) )/ veh.u_max

						if root_p < 0:
							print(f"ERROR")
							print(a)

						tau_hat = root_m * int(root_m > 0) + root_p * int(root_m <= 0)

					veh.ttc_rew = veh.priority * (tau_hat - (veh.exittime -  t_init))

					if veh.ttc_rew > 0:
						print(f"ERROR!!!!!")
						print(f"tau_hat: {tau_hat}, time_to_exit: {(veh.exittime -  t_init)}")
						print(a)

					veh.comb_opt_like_cost += veh.priority * data_file.W_pos * (veh.p_traj[int(math.ceil(data_file.T_r / data_file.dt))-1] - veh.p0)
					
					veh.traffic_eval_func_val = optimal_sol.traf_eval[ln][veh_ind]

					comb_opt_like_cost += veh.comb_opt_like_cost # veh.ttc_rew # optimal_sol.each_veh_cost[ln][veh_ind] 

					if not learn_flag:
						functions.storedata(veh, tr_sim_num, sim_num, train_iter_num)

					coord_set[ln].append(copy.deepcopy(veh))
					rem_ve = prov_set[ln].popleft()
					

				else:
					print("Some vehicle(s) did not cross the intersection!")
					flag_some_did_not_cross_in_incomp_lane[li] = True
					for lin in veh.incomp:
						flag_some_did_not_cross_in_incomp_lane[lin] = True
					
			for vehic in prov_set[ln]:
				vehic.p_traj.append(vehic.coord_init_pos)
				vehic.v_traj.append(vehic.coord_init_vel)
				vehic.t_ser.append(t_init)

				dist_to_cross = (vehic.intsize + vehic.length) - vehic.coord_init_pos

				dist_to_hit_vm = ((vehic.v_max**2) - (vehic.coord_init_vel**2)) / (2 * vehic.u_max)

				if dist_to_cross >= dist_to_hit_vm:
					tau_hat = ((dist_to_cross - dist_to_hit_vm) / (vehic.v_max)) + ((vehic.v_max - vehic.coord_init_vel)/ (vehic.u_max))

				else:
					root_p = (-vehic.coord_init_vel + np.sqrt((vehic.coord_init_vel**2) + (2*vehic.u_max*dist_to_cross)) )/ vehic.u_max
					root_m = (-vehic.coord_init_vel - np.sqrt((vehic.coord_init_vel**2) + (2*vehic.u_max*dist_to_cross)) )/ vehic.u_max
					tau_hat = root_m * int(root_m > 0) + root_p * int(root_m <= 0)
					if root_p < 0:
						print(f"ERROR")
						print(a)

				comb_opt_like_cost -= 5 * data_file.W_pos * (vehic.coord_init_pos - vehic.p0) ** 2

		if data_file.run_coord_on_captured_snap and num_of_vehs_entering_coord_phase:
			return coord_set, result.tot_cost, comb_opt_like_cost, 0, (result.solve_only_computation_time/num_of_vehs_entering_coord_phase), (result.computation_time/num_of_vehs_entering_coord_phase) 
			
		return coord_set, total_cost, comb_opt_like_cost, state, action, True
		
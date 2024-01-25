# this file contains definitions of some general functions required

import os
import math
import time
import numpy as np
import copy
from itertools import permutations
import pickle
from collections import deque
import casadi as cas
import csv
import data_file
import get_ocp_relaxed_traj


def webster(arr):
	lp = data_file.vm / (2*data_file.u_max) + data_file.int_bound/data_file.vm
	sig0 = 1.2
	sfr = 2 * data_file.vm / (sig0 * data_file.L)
	Y = 2 * arr/sfr
	C = (1.5 * lp * 2 + 5)/(1-Y)
	return math.ceil(C/2)


def find_index(this_v, this_time): 
	# returns the index of 'this_time' in time series data of vehicle 'this_v'

	if round(this_time,1) in this_v.t_ser:
		return this_v.t_ser.index(round(this_time,1))
	else: # error!
		#assert not (len(this_v.t_ser)>0) and (this_time <= this_v.t_ser[-1])
		return None


def pres_seper_dist(pre_v, this_v, this_time):
	# returns the present seperation distancebetween vehicle 'this_v' and the vehicle
	# ahead of it on its lane, 'pre_v' at time isntant 'this_time'
	# this is used only in finding initial seperation

	t_ind_this = find_index(this_v, this_time)
	t_ind_pre = find_index(pre_v, this_time)

	if (t_ind_pre == None):
		# print(f"\nlen of prev.u_traj: {len(pre_v.u_traj)}\n")
		if (len(pre_v.u_traj)>0):
			dist0 = - (2 * pre_v.p0) + pre_v.intsize

	else:
		dist0 = pre_v.p_traj[t_ind_pre]	- (this_v.p0)

	if (round(dist0,4) >= 0): # to avoid numenrical errors
		return dist0


def get_follow_dist(pre_v, this_v, this_time):
	# returns the required following distance between vehicle 'this_v' and vehicle 'pre_v'
	# so that rear end safety constraint is satisfied at the time of spawning 'this_v'

	t_ind_this = find_index(this_v, this_time)
	t_ind_pre = find_index(pre_v, this_time)

	if (t_ind_pre == None) and (len(pre_v.p_traj)>0):
		dist0 = 0

	if (t_ind_pre != None) and (t_ind_this == None):
		pre_v0 = pre_v.v_traj[t_ind_pre]
		dist0 = ( ((pre_v0)**2)/ (2*(pre_v.u_min + 0.001))) - ((this_v.v0**2)/(2*(this_v.u_min + 0.001)))
		
	c0 = max(0, dist0)
	foll_dist1 = 1.45*((pre_v.length) + c0)
	return foll_dist1


def breaking_pos_of_veh_at_time(_veh_obj, _time_duration, _init_time_):
	time_index_curr = find_index(_veh_obj, _init_time_)

	if time_index_curr == None:
		_veh_init_pos = _veh_obj.p0
		_veh_init_vel = _veh_obj.v0

	else:
		_veh_init_pos = _veh_obj.p_traj[time_index_curr]
		_veh_init_vel = _veh_obj.v_traj[time_index_curr]

	return _veh_init_pos + (_veh_init_vel * _time_duration) +  (0.5 * _veh_obj.u_min * (_time_duration ** 2))


def check_init_config(_this_veh, _prev_veh, _t_inst): 
	# returns True if the rear-end-safety is satisfied with initial velocity of 
	# '_this_veh' and velocity of '_prev_veh' at time instant '_t_inst'
	# else, returns False

	if _prev_veh == None:
		return True

	d0 = pres_seper_dist(_prev_veh, _this_veh, _t_inst)
	ed1 = get_follow_dist(_prev_veh, _this_veh, _t_inst)


	t_ind_this = find_index(_this_veh, _t_inst)
	t_ind_pre = find_index(_prev_veh, _t_inst)

	if (t_ind_pre == None) and (len(_prev_veh.p_traj) > 0):
		dist0 = 0

	if (t_ind_pre != None) and (t_ind_this == None):

		if (_prev_veh.u_min != _this_veh.u_min):

			_a = 0.5 * (_prev_veh.u_min - _this_veh.u_min)
			_b = (_prev_veh.v_traj[t_ind_pre] - _this_veh.v0)
			_c = _prev_veh.p_traj[t_ind_pre] - _this_veh.p0 - (2*_prev_veh.length)

			_gamma_pre = - _prev_veh.v_traj[t_ind_pre] / _prev_veh.u_min
			_gamma_this = - _this_veh.v0 / _this_veh.u_min

			_p = _gamma_this - (-_b/(2*_a))
			_q = _a*((-_b/(2*_a))**2) + _b*((-_b/(2*_a))) + _c

			if (_c >= 0) and (breaking_pos_of_veh_at_time(_prev_veh, _gamma_pre, _t_inst) - breaking_pos_of_veh_at_time(_this_veh, _gamma_this, _t_inst) >= 2*_prev_veh.length) and \
			((_a <= 0) or (_c >= max(0, _this_veh.v0 - _prev_veh.v_traj[t_ind_pre])*(_this_veh.v0 - _prev_veh.v_traj[t_ind_pre])/(4*_a) )):

				return True

			else:
				return False

		else:
			if (_prev_veh.p_traj[t_ind_pre] - _this_veh.p0 - 2*_prev_veh.length) >= 2*max(0, ((_prev_veh.v_traj[t_ind_pre]**2) - (_this_veh.v0**2))/(2*(_this_veh.u_min))):
				return True
			else:
				return False


	else:
		return False


def check_compat(v1, v2):
	# returns -1 if vehicles 'v1' and 'v2' belong to same lane
	# returns 0 if vehicles 'v1' and 'v2' belong to incompatible lanes
	# else returns 1

	if v1.lane == v2.lane:
		return -1

	elif v2.lane in data_file.incompdict[v1.lane]:
		return 0

	else:
		return 1


def compute_state(_X, _u, _dt):
	# returns the next state when the current state is '_X',
	# step input '_u' with a discrete time step of '_dt' units
	
	##### integration of linear system with matrix exponential #####

	r1 = ((_X[0]) + (_X[1]*round(_dt,1)) + ((round(_dt,1)**2)/2)*_u)
	r2 = (_X[1]) + (round(_dt,1)*_u)
	return cas.vertcat(r1, r2)

	##### integration of linear system with matrix exponential #####
	

def constraint_dynamics(state_opti_var, next_state, opti_class_object):
	# applies one step dynamics constraint for 'opti_class_object'
	# where 'state_opti_var' is the next step state optimization variable
	# and 'next_state' is the computed state update 

	opti_class_object.subject_to(state_opti_var == next_state)


def prov_constraint_vel_max(veh_object, velo_opti_var, posi_opti_var, opti_class_object): 
	# applies constraint on the vehicle velocity optimization variable, 'velo_var'
	# so that it does not enter the intersection

	opti_class_object.subject_to((velo_opti_var**2) <= (2*(veh_object.u_min)*(posi_opti_var + 0.1)))

def constraint_vel_bound(veh_object, velo_opti_var, opti_class_object):
	# applies upper and lower bounds on the vehicle velocity optimization variable, 'velo_opti_var'

	opti_class_object.subject_to(opti_class_object.bounded(veh_object.v_min, velo_opti_var, veh_object.v_max))


def constraint_acc_bound(veh_object, acc_opti_var, opti_class_object):
	# applies upper and lower bounds on the vehicle acceleration optimization variable, 'acc_opti_var'

	opti_class_object.subject_to(opti_class_object.bounded(veh_object.u_min, acc_opti_var, veh_object.u_max)) 

def constraint_init_pos(posi_opti_var, opti_class_object, init_pos):
	# applies initial position constraint for vehicle position optimization variable, 'posi_opti_var'

	opti_class_object.subject_to(posi_opti_var[0,0] == init_pos)

def constraint_init_vel(velo_opti_var, opti_class_object, init_vel): 
	# applies initial velocity constraint for vehicle velocity optimization variable, 'velo_opti_var'

	opti_class_object.subject_to(velo_opti_var[1,0] == init_vel)
	

def prov_constraint_rear_end_safety(veh_object, prev_veh, init_t_inst, posi_var, vel_var, opti_class_object):
	# applies rear-end-safety constraint on position and velocity optimization variables of a vehicle
	# depending on the postion and velocity trajectories of the vehicle ahead on its lane, 'prev_veh'
	#return
	ind_prev = find_index(prev_veh, round(init_t_inst,1))
	if ind_prev == None:
		return

	time_since_last_planned = 0
	for k in range(posi_var.size2()):
			
		if len(prev_veh.t_ser) > (ind_prev + k):

			opti_class_object.subject_to(prev_veh.p_traj[ind_prev + k] - posi_var[k] -  prev_veh.length >= 0)
			opti_class_object.subject_to(prev_veh.p_traj[ind_prev + k] - posi_var[k] -  prev_veh.length >= ((prev_veh.v_traj[ind_prev + k]**2) - (vel_var[k]**2))/(2*veh_object.u_min) )



		else:
			time_since_last_planned += round(data_file.dt, 1)

			prev_veh_pos_now = prev_veh.p_traj[-1] + (prev_veh.v_traj[-1]*time_since_last_planned)
			prev_veh_vel_now = prev_veh.v_traj[-1]

			opti_class_object.subject_to(prev_veh_pos_now - posi_var[k] - prev_veh.length >= 0)
			opti_class_object.subject_to(prev_veh_pos_now - posi_var[k] - prev_veh.length >= ((prev_veh_vel_now**2) - (vel_var[k]**2))/(2*veh_object.u_min) )


def coord_constraint_rear_end_safety(veh_object, prev_veh, init_t_inst, posi_var, vel_var, opti_class_object):
	# applies rear-end-safety constraint on position and velocity optimization variables of a vehicle
	# depending on the postion and velocity trajectories of the vehicle ahead on its lane, 'prev_veh'
	#return
	ind_prev = find_index(prev_veh, round(init_t_inst,1))

	if ind_prev == None:
		return
	
	time_since_last_planned = 0
	for k in range(posi_var.size2()):
			
		if len(prev_veh.t_ser) > (ind_prev + k): 

			opti_class_object.subject_to(prev_veh.p_traj[ind_prev + k] - posi_var[k] - prev_veh.length >= 0)
			opti_class_object.subject_to(prev_veh.p_traj[ind_prev + k] - posi_var[k] - prev_veh.length >= ((prev_veh.v_traj[ind_prev + k]**2) - (vel_var[k]**2))/(2*veh_object.u_min) )
			
		else:
			time_since_last_planned += round(data_file.dt, 1)

			prev_veh_pos_now = prev_veh.p_traj[-1] + (prev_veh.v_traj[-1]*time_since_last_planned)
			prev_veh_vel_now = prev_veh.v_traj[-1]

			opti_class_object.subject_to(prev_veh_pos_now - posi_var[k] - prev_veh.length >= 0)
			opti_class_object.subject_to(prev_veh_pos_now - posi_var[k] - prev_veh.length >= ((prev_veh_vel_now**2) - (vel_var[k]**2))/(2*veh_object.u_min) )
				

				

def constraint_waiting_time(_wait_time, pos_var, opti_class_object):
	st = int(math.ceil(((round(_wait_time,1))/round(data_file.dt,1))))
	if st > 0:
		if pos_var.size2() > st:
			opti_class_object.subject_to(pos_var[st] <= 0)

		else:
			opti_class_object.subject_to(pos_var[-1] <= 0)



def get_num_of_objects(td_list):
	total_num_objects = 0
	for li in data_file.lanes:
		total_num_objects += len(td_list[li])

	return total_num_objects


def get_set_f(_p_set):
	F = [deque([])for _ in data_file.lanes]
	for l in data_file.lanes:
		if len(_p_set[l]) > 0:
			F[l].append(_p_set[l][0])

	return F
			

def storedata(veh_object, train_sim_num, _sim, _train_iter_num):    
    m = {}
    m[veh_object.id] = veh_object
    
    if data_file.rl_flag and (not data_file.run_coord_on_captured_snap):
    	if data_file.used_heuristic == None:
    		if len(data_file.arr_rates_to_simulate) > 1:
    			dbfile = open(f'../data/arr_{veh_object.arr}/test_homo_stream/train_sim_{train_sim_num}/train_iter_{_train_iter_num}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
    		else:	
    			dbfile = open(f'../data/arr_{data_file.arr_rates_to_simulate[0]}/test_homo_stream/train_sim_{train_sim_num}/train_iter_{_train_iter_num}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')


    	elif len(data_file.arr_rates_to_simulate) == 1:
    		dbfile = open(f'../data/{data_file.used_heuristic}/arr_{data_file.arr_rates_to_simulate[0]}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')

    	else:
    		dbfile = open(f'../data/{data_file.used_heuristic}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')
    
    else:
    	dbfile = open(f'../data/{data_file.algo_option}/arr_{veh_object.arr}/pickobj_sim_{_sim}/'+str(veh_object.id), 'wb')


    pickle.dump(m, dbfile)
    dbfile.close()


def get_feasible_schedules(__prov_set):

	def in_order_combinations(*lists):
		lists = list(filter(len, lists))

		if len(lists) == 0:
			yield []

		for lst in lists:
			element = lst.pop()
			for combination in in_order_combinations(*lists):
				yield combination + [element]
			lst.append(element)

	feasible_perms = [_ for _ in in_order_combinations(*__prov_set)]

	return feasible_perms


def check_feasibility(state, inp, veh_obj, prev_veh_obj, init_pos, init_vel, init_time, opti_class_object, debug_flag):

	if debug_flag == 0:

		init_pos_flag = 1
		init_vel_flag = 1
		#### checking initial conditions ####
		if round(opti_class_object.value(state[0,0]),4) != round(init_pos,4):
			print("initial position violated! constraint:", init_pos, "solution value:", opti_class_object.value(state[0,0]))
			init_pos_flag = 0
		
		else:
			pass



		if round(opti_class_object.value(state[1,0]),4) != round(init_vel,4):
			print("initial velocity violated! constraint:", init_vel, "solution value:", opti_class_object.value(state[1,0]))
			init_vel_flag = 0

		else:
			pass


		#### checking velocity and acceleration bounds ####
		bound_flag = 1
		for k in range(state.size2()):
			if (round(opti_class_object.value(state[1,k]),4) > veh_obj.v_max) or (round(opti_class_object.value(state[1,k]),4) < 0):
				print("velocity bound violated!", round(opti_class_object.value(state[1,k]),4))
				bound_flag = 0
				break
			
			if (round(opti_class_object.value(inp[k]),4) > veh_obj.u_max) or (round(opti_class_object.value(inp[k]),4) < veh_obj.u_min):
				print("acceleration bound violated!")
				bound_flag = 0
				break

		if bound_flag == 1:
			pass

		#### checking rear-end safety ####
		rear_end_flag = 1

		if prev_veh_obj != None:
			ind_prev = find_index(prev_veh_obj, init_time)
			min_difference =  -80
			for k in range(state.size2()):
				bool_for_index = ((ind_prev + k) < len(prev_veh_obj.t_ser))
				
				if bool_for_index:
					bool_for_constraint_with_prev_traj = (round(opti_class_object.value(state[0,k]), 4) <= round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) )- (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))

					min_difference = max(min_difference, (opti_class_object.value(state[0,k])) - (prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )))))

					if (not bool_for_constraint_with_prev_traj):
						print("rear-end-safety constraint violated!1", round(opti_class_object.value(state[0,k]), 4), round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))
						rear_end_flag = 0
						break


				else:
					time_since_last_planned = round(((init_time + (k*data_file.dt)) - prev_veh_obj.t_ser[-1]), 1) 
					bool_for_constraint_without_prev_traj = (round(opti_class_object.value(state[0,k]), 4) <= round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))

					min_difference = max(min_difference, (opti_class_object.value(state[0,k])) - (((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )) ))))

					if (not bool_for_constraint_without_prev_traj):
						print("rear-end-safety constraint violated!2", round(opti_class_object.value(state[0,k]), 4), round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))
						rear_end_flag = 0
						break

			with open(f"./data/min_constraint_diff.csv", "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows([[min_difference]])

		if rear_end_flag == 1:
			print("rear-end-safety constraint satisfied")

		#### checking for intersection safety ####

		inter_safety_flag = 1
		min_wait_time_index = int(math.ceil(((round(veh_obj.feat_min_wait_time,1))/round(data_file.dt,1))))

		for k in range(min_wait_time_index+1):
			if round(opti_class_object.value(state[0,k]),4) > 0:
				print("minimum wait time constratint violated!", min_wait_time_index, k, opti_class_object.value(state[0,k]))
				inter_safety_flag = 0
				break

		if inter_safety_flag == 1:
			pass


		


	if debug_flag == 1:

		init_pos_flag = 1
		init_vel_flag = 1
		#### checking initial conditions ####
		if round(opti_class_object.value(state[0,0]),4) != round(init_pos,4):
			print("initial position violated! constraint:", init_pos, "solution value:", opti_class_object.debug.value(state[0,0]))
			init_pos_flag = 0
		
		else:
			pass


		if round(opti_class_object.value(state[1,0]),4) != round(init_vel,4):
			print("initial velocity violated! constraint:", init_vel, "solution value:", opti_class_object.debug.value(state[1,0]))
			init_vel_flag = 0

		else:
			pass


		#### checking velocity and acceleration bounds ####
		bound_flag = 1
		for k in range(state.size2()):
			if (round(opti_class_object.debug.value(state[1,k]),4) > veh_obj.v_max) or (round(opti_class_object.debug.value(state[1,k]),4) < 0):
				print("velocity bound violated!")
				bound_flag = 0
				break
			
			if (round(opti_class_object.debug.value(inp[k]),4) > data_file.u_max) or (round(opti_class_object.debug.value(inp[k]),4) < veh_obj.u_min):
				print("acceleration bound violated!")
				bound_flag = 0
				break

		if bound_flag == 1:
			pass

		#### checking rear-end safety ####
		min_difference = -80
		rear_end_flag = 1
		if prev_veh_obj != None:
			ind_prev = find_index(prev_veh_obj, init_time)
			for k in range(state.size2()):
				bool_for_index = ((ind_prev + k) < len(prev_veh_obj.t_ser))
				if bool_for_index:
					bool_for_constraint_with_prev_traj = (round(opti_class_object.debug.value(state[0,k]), 4) <= round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))

					min_difference = max(min_difference, (opti_class_object.debug.value(state[0,k])) - (prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )))))

					if (not bool_for_constraint_with_prev_traj):
						print("rear-end-safety constraint violated1!", round(opti_class_object.debug.value(state[0,k]), 4), round(prev_veh_obj.p_traj[ind_prev + k] - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[ind_prev+k]**2) / (2*prev_veh_obj.u_min) ) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) ))),4))
						rear_end_flag = 0


				else:
					time_since_last_planned = round(((init_time + (k*data_file.dt)) - prev_veh_obj.t_ser[-1]), 1) 
					bool_for_constraint_without_prev_traj = (round(opti_class_object.debug.value(state[0,k]), 4) <= round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min) )- (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))
					
					min_difference = max(min_difference, (opti_class_object.debug.value(state[0,k])) - (((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min) )- (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )) ))))

					if (not bool_for_constraint_without_prev_traj):
						print("rear-end-safety constraint violated2!", round(opti_class_object.debug.value(state[0,k]), 4), round(((prev_veh_obj.p_traj[-1] + ((prev_veh_obj.v_traj[-1])*time_since_last_planned)) - prev_veh_obj.length - max(0, ((( ((prev_veh_obj.v_traj[-1]**2) / (2*prev_veh_obj.u_min)) - (opti_class_object.debug.value(state[1,k])**2)) / (2*veh_obj.u_min) )) )) ,4))
						rear_end_flag = 0

			with open(f"./data/min_constraint_diff_exceeded.csv", "a", newline="") as f:
				writer = csv.writer(f)
				writer.writerows([[min_difference]])

		if rear_end_flag == 1:
			pass

		#### checking for intersection safety ####

		inter_safety_flag = 1
		min_wait_time_index = int(math.ceil(((round(veh_obj.feat_min_wait_time,1))/round(data_file.dt,1))))

		for k in range(min_wait_time_index):
			if round(opti_class_object.debug.value(state[0,k]),4) > 0:
				print("minimum wait time constratint violated!", min_wait_time_index, k, opti_class_object.debug.value(state[0,k]))
				inter_safety_flag = 0
				break

		if inter_safety_flag == 1:
			pass
		

	assert not ((init_pos_flag == 0) or (init_vel_flag == 0) or (bound_flag == 0) or (rear_end_flag == 0) or (inter_safety_flag == 0))




def check_feasibility_Vc_Vs(V_c, V_s, schedu, sol_pos, sol_vel, sol_acc, curr_time, init_c, opti_steps, sch_count, stri):

	init_pos_flag = 0
	init_vel_flag = 0
	vel_bound_flag = 0
	acc_bound_flag = 0
	rear_end_flag = 0
	inter_safety_flag = 0


	for _lan in data_file.lanes:
		
		### checking initial position constraint ###
		if not init_pos_flag:
			veh_pos_index = 0
			for veh_pos_ in sol_pos[_lan]:
				if not (round(veh_pos_[0], 4) == round(init_c[_lan][veh_pos_index][0], 4)):
					init_pos_flag = 1
					pos_a = round(veh_pos_[0], 4)
					pos_b = round(init_c[_lan][veh_pos_index][0], 4)
					break
				veh_pos_index += 1

		### checking initial velocity constraint ###
		if not init_vel_flag:
			veh_vel_index = 0			
			for veh_vel_ in sol_vel[_lan]:
				if not (round(veh_vel_[0], 4) == round(init_c[_lan][veh_vel_index][1], 4)):
					init_vel_flag = 1
					vel_a = round(veh_vel_[0], 4)
					vel_b = round(init_c[_lan][veh_vel_index][1], 4)
					break
				veh_vel_index += 1



		### checking velocity bounds ###
		if not vel_bound_flag:
			for ind, veh_vel_ in enumerate(sol_vel[_lan]):
				veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
				current_vehicle = veh_in_lane[ind]
				for vel_ in veh_vel_:
					if round(vel_, 4) < current_vehicle.v_min:
						vel_bound_flag = 1
						break

					if round(vel_, 4) > current_vehicle.v_max:
						vel_bound_flag = 1
						break

				if vel_bound_flag:
					break

		### checking acceleration bounds ###
		if not acc_bound_flag:
			for ind, veh_acc_ in enumerate(sol_acc[_lan]):
				veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
				current_vehicle = veh_in_lane[ind]
				for acc_ in veh_acc_:
					if round(acc_, 4) < current_vehicle.u_min:
						acc_bound_flag = 1
						break

					if round(acc_, 4) > current_vehicle.u_max:
						acc_bound_flag = 1
						break

				if acc_bound_flag:
					break



		### checking rear-end-safety ###
		if not rear_end_flag:
			for veh_index_rear_end in range(len(V_c[_lan])):
				veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
				current_vehicle = veh_in_lane[veh_index_rear_end]
				if (veh_index_rear_end == 0) and (len(V_s[_lan]) > 0):
					prev_veh_ = V_s[_lan][-1]
					ind_prev = find_index(prev_veh_, round(curr_time,1))
					for time_index in range(1, len(sol_pos[_lan][veh_index_rear_end])):

						if len(V_s[_lan][-1].t_ser) > (ind_prev + time_index): 
							if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (prev_veh_.p_traj[ind_prev + time_index] - (prev_veh_.length)) , 4) > 0.0001):
								rear_end_flag = 1
								print("here1")
								break

							if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (prev_veh_.p_traj[ind_prev + time_index] - ((prev_veh_.length) + ((( ((prev_veh_.v_traj[ind_prev + time_index]**2) / (2*prev_veh_.u_min) )- (sol_vel[_lan][veh_index_rear_end][time_index]**2)) / (2*((current_vehicle.u_min) ) ))))), 4) > 0.0001 ):
								rear_end_flag = 1
								print("here2")
								break

						else:
							time_since_last_planned = round(((ind_prev + time_index + 1 - len(prev_veh_.t_ser))*data_file.dt), 1)
							if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (prev_veh_.p_traj[-1] + (prev_veh_.v_traj[-1]*time_since_last_planned) - (prev_veh_.length)), 4) > 0.0001 ):
								rear_end_flag = 1
								print("here3")
								break

							if (round(sol_pos[_lan][veh_index_rear_end][time_index] -(prev_veh_.p_traj[-1] + (prev_veh_.v_traj[-1]*time_since_last_planned) - ((prev_veh_.length) + ((((prev_veh_.v_traj[-1]**2) / (2*prev_veh_.u_min) )- (sol_vel[_lan][veh_index_rear_end][time_index]**2)) / (2*((current_vehicle.u_min) ) )))), 4) > 0.0001 ):
								rear_end_flag = 1
								print("here4")
								break

						if rear_end_flag:
							break

				elif ( not (veh_index_rear_end == 0)):

					veh_in_lane = list(filter(lambda x: x.lane==_lan, schedu))
					previous_vehicle = veh_in_lane[veh_index_rear_end-1]

					for time_index in range(1, len(sol_pos[_lan][veh_index_rear_end])):

						if (round(sol_pos[_lan][veh_index_rear_end][time_index] - ((sol_pos[_lan][veh_index_rear_end-1][time_index] - (previous_vehicle.length))),4) > 0.0001):
							rear_end_flag = 1
							print(sol_pos[_lan][veh_index_rear_end][time_index])

							print("\n")
							print(sol_pos[_lan][veh_index_rear_end])
							print("\n")
							print(sol_pos[_lan][veh_index_rear_end-1][time_index])
							print("\n")
							print(_lan, veh_index_rear_end)
							print("here5")
							break

						if (round(sol_pos[_lan][veh_index_rear_end][time_index] - (sol_pos[_lan][veh_index_rear_end-1][time_index] - ((previous_vehicle.length) + (((( (sol_vel[_lan][veh_index_rear_end-1][time_index]**2) /(2*previous_vehicle.u_min) ) - (sol_vel[_lan][veh_index_rear_end][time_index]**2)) / (2*((current_vehicle.u_min) ) ))))), 4) > 0.0001):
							rear_end_flag = 1
							print("here6")
							break

				if rear_end_flag:
					break


		if init_pos_flag and init_vel_flag and vel_bound_flag and acc_bound_flag and rear_end_flag:
			break



	### intersection safety ###

	if not inter_safety_flag:
		for time_index in range(opti_steps):
			for _lan in data_file.lanes:
				if len(data_file.incompdict[_lan]) > 0:
					for veh_index_int_safety in range(len(V_c[_lan])):
						if (sol_pos[_lan][veh_index_int_safety][time_index] > 0) and (sol_pos[_lan][veh_index_int_safety][time_index] < V_c[_lan][veh_index_int_safety].intsize):
							for incomp_lan in V_c[_lan][veh_index_int_safety].incomp:

								if len(V_s[incomp_lan]) > 0:
									incomp_veh_t_ind = find_index(V_s[incomp_lan][-1], curr_time)

									if (not (incomp_veh_t_ind == None)) and (incomp_veh_t_ind + time_index < len(V_s[incomp_lan][-1].t_ser)):

										if (V_s[incomp_lan][-1].p_traj[incomp_veh_t_ind + time_index] > 0) and (V_s[incomp_lan][-1].p_traj[incomp_veh_t_ind + time_index] < V_s[incomp_lan][-1].intsize):
											 inter_safety_flag = 1
											 break

						if inter_safety_flag:
							break

					if inter_safety_flag:
						break

			if inter_safety_flag:
				break


	if init_pos_flag:
		print("inital position constraint violated!", sch_count, stri, )

	if init_vel_flag:
		print("inital velocity constraint violated!", sch_count, stri)

	if vel_bound_flag:
		print("velocity bound constraint violated!", sch_count, stri)

	if acc_bound_flag:
		print("acceleration bound constraint violated!", sch_count, stri)

	if rear_end_flag:
		print("rear-end-safety constraint violated!", sch_count, stri)

	if inter_safety_flag:
		print("intersection safety constraint violated!", sch_count, stri)


	if init_pos_flag or init_vel_flag or vel_bound_flag or acc_bound_flag or rear_end_flag or inter_safety_flag:
		print([init_pos_flag, init_vel_flag, vel_bound_flag, acc_bound_flag, rear_end_flag, inter_safety_flag], sch_count, stri)
		if init_pos_flag:
			print(pos_a, pos_b)
		if init_vel_flag:
			print(vel_a, vel_b)

		m = {}
		m['vc'] = V_c
		m['vs'] = V_s
		m['t'] = curr_time
		m['sched'] = schedu
		num_bad_seq_till_now = len(list(os.listdir(f'./data/compare_files/arr_{data_file.arr_rate}/bad_seqs/')))
		dbfile = open(f"./data/compare_files/arr_{data_file.arr_rate}/bad_seqs/seq_{num_bad_seq_till_now}", 'wb')
		pickle.dump(m, dbfile)
		dbfile.close()
		

def get_ocp_prece_indices(_prov_set_, _coord_set_, _t_init):

	time_init_ = time.time()

	_prov_set_copy_ = copy.deepcopy(_prov_set_)
	_coord_set_copy_ = copy.deepcopy(_coord_set_)

	_updated_prov_set_ = [deque([]) for _ in range(len(data_file.lanes))]

	_init_conds = [deque([]) for _ in range(len(data_file.lanes))]
	
	for _lane_ind_ in range(len(_prov_set_copy_)):
		for vehic_ind in range(len(_prov_set_copy_[_lane_ind_])):
			_init_pos = _prov_set_copy_[_lane_ind_][vehic_ind].coord_init_pos
			_init_vel = _prov_set_copy_[_lane_ind_][vehic_ind].coord_init_vel
			_init_conds[_lane_ind_].append([_init_pos, _init_vel])

	temp_obj = get_ocp_relaxed_traj.ocp_relaxed_opt(_prov_set_copy_, _coord_set_copy_, _t_init, _init_conds)

	_updated_prov_set_ = copy.deepcopy(temp_obj.updated_prov_set)

	for _lane_ind_ in range(len(_updated_prov_set_)):
		for vehic_ind in range(len(_updated_prov_set_[_lane_ind_])):

			ent_time_p_val = list(filter(lambda posi: posi > 0, _updated_prov_set_[_lane_ind_][vehic_ind].p_traj))[0]
			ent_time_index = list(_updated_prov_set_[_lane_ind_][vehic_ind].p_traj).index(ent_time_p_val)

			_updated_prov_set_[_lane_ind_][vehic_ind].entry_time = round(_updated_prov_set_[_lane_ind_][vehic_ind].t_ser[ent_time_index], 1)

			exit_time_p_val = list(filter(lambda posi: posi > (_updated_prov_set_[_lane_ind_][vehic_ind].intsize + _updated_prov_set_[_lane_ind_][vehic_ind].length), _updated_prov_set_[_lane_ind_][vehic_ind].p_traj))[0]
			exit_time_index = list(_updated_prov_set_[_lane_ind_][vehic_ind].p_traj).index(exit_time_p_val) + 1

			_updated_prov_set_[_lane_ind_][vehic_ind].exit_time = round(_updated_prov_set_[_lane_ind_][vehic_ind].t_ser[exit_time_index], 1)

	temp_list = []

	for lane in _updated_prov_set_:
		for _veh_ in lane:
			temp_list.append(_veh_)

	temp_list.sort(key=lambda x:x.entry_time)
	N_i = copy.deepcopy(temp_list)
	temp_list.sort(key=lambda x:x.exit_time)
	N_o = copy.deepcopy(temp_list)

	in_while_flag = True

	while in_while_flag:
		N_i_temp = copy.deepcopy(N_i)
		in_while_flag = False
		item_ind = 0
		for item_ind in range(len(temp_list[:-1])):
			item = N_i[item_ind]
			next_item_ind = item_ind + 1
			next_item = N_i[next_item_ind]
			if next_item.lane not in data_file.incompdict[item.lane]:
				for temp_item in N_o:
					if temp_item.id == item.id:
						break
					elif temp_item.id == next_item.id:
						N_i_temp[item_ind] = next_item
						N_i_temp[item_ind+1] = item
						in_while_flag = True
						N_i = copy.deepcopy(N_i_temp)
						break
			item_ind += 1
	
	temp_var_ind = 0

	for _temp_veh in N_i_temp:
		for _veh_ind_, _veh_ in enumerate(_prov_set_copy_[_temp_veh.lane]):
			if _veh_.id == _temp_veh.id:
				temp_var_ind += 1
				_veh_.priority_index = -temp_var_ind
				_updated_prov_set_[_temp_veh.lane][_veh_ind_].priority_index = -temp_var_ind

	for l_ind_, __lane__ in enumerate(_prov_set_):
		for __veh_ind, __veh in enumerate(__lane__):
			if _updated_prov_set_[l_ind_][__veh_ind].priority_index == None:
				print(f"bad veh id: {__veh.id}, arr: {__veh.arr}")
				print(f"bad veh id: {__veh.id}, arr: {__veh.arr}")
				print(a)

	return _updated_prov_set_



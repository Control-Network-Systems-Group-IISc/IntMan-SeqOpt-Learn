import os
import casadi as cas 
import math
import copy
import numpy as np
import csv
import pickle
import time
import math
import data_file
import functions

class coord_opti_class_seq_opt():
	def __init__(self, max_iter, _sched_count, _sched, _prov_set, _coord_set, curr_time, _init_conds):
		self.disc_factor = 1

		self.max_iter_exceeded = False
		self.next_max_iter = max_iter

		self.prov_set_ = copy.deepcopy(_prov_set)
		self.coord_set_ = copy.deepcopy(_coord_set)

		self.tot_cost = 0
		# self.X = []
		# self.acc = []
		self.jerk = [[] for _ in _sched]
		self.pos = [[] for _ in data_file.lanes]
		self.vel = [[] for _ in data_file.lanes]
		self.acc_u = [[] for _ in data_file.lanes]
		self.sol = [[] for _ in data_file.lanes]
		self.obj_fun = 0
		self.acces_var = []

		self.each_veh_cost = [[]]*len(data_file.lanes)
		self.traf_eval = [[]]*len(data_file.lanes)

		self.steps = int(math.ceil(data_file.T_sc/data_file.dt))

		large_pos_value = 100

		self.obj_fun_for_each_veh = [0]*len(_sched)

		self.traffic_eval_fun = [0]*len(_sched)

		wait_time_from_prev_iter = [0 for _ in data_file.lanes]

		self.solve_only_computation_time = 0

		comp_init_time = time.time()

		for j in range(len(_sched)):

			veh = _sched[j]

			opti = cas.Opti()

			self.obj_fun = opti.variable()

			self.X = opti.variable(2, self.steps + 1)
			self.acc = opti.variable(1, self.steps + 1)

			opti.subject_to(self.obj_fun == sum([(veh.priority * data_file.W_pos * self.X[1,i] * data_file.dt) for i in range(self.steps)]))

			### construction of constraints ###

			### discrete time dynamics ###
			
			for k in range(self.steps):
				x_next = functions.compute_state(self.X[:,k], self.acc[k], data_file.dt)
				functions.constraint_dynamics(self.X[:,k+1], x_next, opti)


			functions.constraint_vel_bound(veh, self.X[1,:], opti)
			functions.constraint_acc_bound(veh, self.acc[:], opti)

			_pre_v = None
			if len(self.coord_set_[veh.lane]) > 0:
				_pre_v = self.coord_set_[veh.lane][-1]

			if _pre_v != None:
				functions.coord_constraint_rear_end_safety(veh, _pre_v, curr_time, self.X[0,:], self.X[1,:], opti)
				pass


			ind = [x.id for x in _prov_set[veh.lane]].index(veh.id)
			_init_pos0 = copy.deepcopy(veh.coord_init_pos)
			_init_vel0 = copy.deepcopy(veh.coord_init_vel)

			### initital conditions ###
			functions.constraint_init_pos(self.X, opti, _init_pos0)
			functions.constraint_init_vel(self.X, opti, _init_vel0)

			wait_time = 0
			incomp_lane_list = np.zeros((len(veh.incomp),1))
			for lan in veh.incomp:
				for v_o in self.coord_set_[lan]: # use something like incomp_lane_list to reduce computations
					if v_o.exittime > curr_time:
						wait_time = round(max(wait_time, (v_o.exittime - curr_time), wait_time_from_prev_iter[v_o.lane]),1)
			functions.constraint_waiting_time(wait_time, self.X[0,:], opti)

			opti.minimize(-self.obj_fun)
			p = {"print_time": False, "ipopt.print_level": 0}#,"ipopt.bound_relax_factor": 0, "ipopt.honor_original_bounds": "yes", "ipopt.constr_viol_tol": 10**-3}#, "ipopt.max_iter": max_iter, "ipopt.constr_viol_tol": 10**-4}#, "ipopt.constr_viol_tol": 10**-3, "ipopt.acceptable_tol": 10**-4}#, "ipopt.tol": 10**-4}
			opti.solver('ipopt', p)

			try:

				solve_init_time = time.time()
				temp_s = opti.solve()
				time_solve_duration = time.time() - solve_init_time
				self.solve_only_computation_time += time_solve_duration

				veh = _sched[j]
				self.pos[veh.lane].append(temp_s.value(self.X[0,:]))
				self.vel[veh.lane].append(temp_s.value(self.X[1,:]))
				self.acc_u[veh.lane].append(temp_s.value(self.acc[:]))
				self.each_veh_cost[veh.lane].append(temp_s.value(self.obj_fun_for_each_veh[j]))#[temp_s.value(veh_cost_val) for veh_cost_val in self.obj_fun_for_each_veh]

				self.traf_eval[veh.lane].append(temp_s.value(self.traffic_eval_fun[j]))


				self.tot_cost += temp_s.value(self.obj_fun)
			

			except:
				if opti.return_status() == "Maximum_Iterations_Exceeded":
					print(f"Maximum_Iterations_Exceeded_{_sched_count}", opti.stats()["return_status"])
					self.max_iter_exceeded = True
					self.next_max_iter = 2*max_iter

					temp_s = opti

					veh = _sched[j]
					self.pos[veh.lane].append(temp_s.debug.value(self.X[0,:]))
					self.vel[veh.lane].append(temp_s.debug.value(self.X[1,:]))
					self.acc_u[veh.lane].append(temp_s.debug.value(self.acc[:]))

					self.each_veh_cost[veh.lane].append(temp_s.debug.value(self.obj_fun_for_each_veh[j]))#[temp_s.value(veh_cost_val) for veh_cost_val in self.obj_fun_for_each_veh]

					self.traf_eval[veh.lane].append(temp_s.debug.value(self.traffic_eval_fun[j]))

					self.tot_cost += temp_s.debug.value(self.obj_fun)

					if opti.debug.show_infeasibilities(10**(-3)) == None:
						pass

					else:
						print(a)
						break

				
				else:
					self.max_iter_exceeded = True
					
					# print(opti.return_status())

					if opti.debug.show_infeasibilities(10**(-3)) == None:
						temp_s = opti
						veh = _sched[j]
						self.pos[veh.lane].append(temp_s.debug.value(self.X[0,:]))
						self.vel[veh.lane].append(temp_s.debug.value(self.X[1,:]))
						self.acc_u[veh.lane].append(temp_s.debug.value(self.acc[:]))
						self.each_veh_cost[veh.lane].append(temp_s.debug.value(self.obj_fun_for_each_veh[j]))#[temp_s.value(veh_cost_val) for veh_cost_val in self.obj_fun_for_each_veh]
						self.traf_eval[veh.lane].append(temp_s.debug.value(self.traffic_eval_fun[j]))
						self.tot_cost += temp_s.debug.value(self.obj_fun)

					else:	
						print(a)
						break


				
			try:
				wait_time_pos = list(filter(lambda posi: posi > (veh.intsize + veh.length), self.pos[veh.lane][-1]))[0]

				wait_time_ind = list(self.pos[veh.lane][-1]).index(wait_time_pos)
				wait_time_from_prev_iter[veh.lane] = round((wait_time_ind * data_file.dt), 1)

				veh.exittime = round(curr_time+(wait_time_ind * data_file.dt), 1)

				self.each_veh_cost[veh.lane].append(temp_s.value(self.obj_fun_for_each_veh[j]))
				self.traf_eval[veh.lane].append(temp_s.value(self.traffic_eval_fun[j]))

				self.prov_set_[veh.lane].popleft()

				for n in range(self.steps):
					veh.p_traj.append(self.pos[veh.lane][-1][n])
					veh.v_traj.append(self.vel[veh.lane][-1][n])
					veh.u_traj.append(self.acc_u[veh.lane][-1][n])
					veh.t_ser.append(round(curr_time + (n * data_file.dt), 1))
				self.coord_set_[veh.lane].append(veh)

			except:

				if len(self.pos[veh.lane]) > 1:
					self.pos[veh.lane] = self.pos[veh.lane][:-1]
					self.vel[veh.lane] = self.vel[veh.lane][:-1]
					self.acc_u[veh.lane] = self.acc_u[veh.lane][:-1]

				else:
					self.pos[veh.lane] = []
					self.vel[veh.lane] = []
					self.acc_u[veh.lane] = []

				break

			if _pre_v != None:
				t_init_for_v = functions.find_index(_pre_v, veh.sp_t)
				for t_ind_for_v in range(len(veh.p_traj)):
					if t_init_for_v + t_ind_for_v >= len(_pre_v.p_traj):
						break

					if round(_pre_v.p_traj[t_init_for_v + t_ind_for_v] - veh.p_traj[t_ind_for_v] - _pre_v.length, 3) < -0.005:
						print("ERROR!!!!!")
						print(a)

		self.computation_time = time.time() - comp_init_time



class coord_opti_class_combined_opt():
	def __init__(self, max_iter, _sched_count, _sched, _prov_set, _coord_set, curr_time, _init_conds):

		opti = cas.Opti()

		self.disc_factor = 1

		self.max_iter_exceeded = False
		self.next_max_iter = max_iter

		self.prov_set_ = copy.deepcopy(_prov_set)
		self.coord_set_ = copy.deepcopy(_coord_set)

		self.tot_cost = 0
		self.X = []
		self.acc = []
		self.jerk = [[] for _ in _sched]
		self.pos = [[] for _ in data_file.lanes]
		self.vel = [[] for _ in data_file.lanes]
		self.acc_u = [[] for _ in data_file.lanes]
		self.sol = [[] for _ in data_file.lanes]
		self.obj_fun = 0
		self.acces_var = []

		self.each_veh_cost = [[]]*len(data_file.lanes)
		self.traf_eval = [[]]*len(data_file.lanes)

		self.steps = int(math.ceil(data_file.T_sc/data_file.dt))

		large_pos_value = 100
		for k in range(len(_sched)):
			veh = _sched[k]
			self.acces_var.append(opti.parameter(len(veh.incomp), self.steps + 1))
			large_pos_value = max(large_pos_value, 10 + (veh.v_max*data_file.T_sc) + (0.5 * veh.u_max * data_file.T_sc**2))

		for j in range(len(_sched)):
			self.X.append(opti.variable(2, self.steps + 1))
			self.acc.append(opti.variable(1, self.steps + 1))

			for i in range(self.steps):
				self.jerk[j].append((self.acc[j][i+1] - self.acc[j][i]) / data_file.dt)

		self.obj_fun_for_each_veh = [0]*len(_sched)

		self.traffic_eval_fun = [0]*len(_sched)

		wait_time_from_prev_iter = [0 for _ in data_file.lanes]

		self.solve_only_computation_time = 0

		comp_init_time = time.time()

		self.obj_fun = 0

		for j in range(len(_sched)):

			veh = _sched[j]
			
			for i in range(self.steps):
				self.obj_fun += veh.priority * data_file.W_pos * self.X[j][1,i] * data_file.dt


			### construction of constraints ###

			### discrete time dynamics ###

			for k in range(self.steps):
				x_next = functions.compute_state(self.X[j][:,k], self.acc[j][k], data_file.dt)
				functions.constraint_dynamics(self.X[j][:,k+1], x_next, opti)


			functions.constraint_vel_bound(veh, self.X[j][1,:], opti)
			functions.constraint_acc_bound(veh, self.acc[j][:], opti)


			_pre_v = None
			skip_flag = 0
			if j > 0:
				itr_var_list = list(range(j))
				itr_var_list.reverse()
				for i in itr_var_list:
					if veh.lane == _sched[i].lane:

						for k in range(self.steps):
						
							opti.subject_to(self.X[j][0,k] <= (self.X[i][0,k] - _sched[i].length))
							opti.subject_to(self.X[j][0,k] <= (self.X[i][0,k] - _sched[i].length - (((self.X[i][1,k]**2) / (2* (_sched[i].u_min))) - (((self.X[j][1,k]**2))/ (2* (veh.u_min))))  ))

						skip_flag = 1
						break

			

			if (skip_flag == 0) and (len(self.coord_set_[veh.lane]) > 0):
				_pre_v = self.coord_set_[veh.lane][-1]

			if _pre_v != None:
				functions.coord_constraint_rear_end_safety(veh, _pre_v, curr_time, self.X[j][0,:], self.X[j][1,:], opti)
				pass


			ind = [x.id for x in self.prov_set_[veh.lane]].index(veh.id)
			_init_pos0 = copy.deepcopy(veh.coord_init_pos)
			_init_vel0 = copy.deepcopy(veh.coord_init_vel)

			### initital conditions ###
			functions.constraint_init_pos(self.X[j], opti, _init_pos0)
			functions.constraint_init_vel(self.X[j], opti, _init_vel0)

			wait_time = 0
			incomp_lane_list = np.zeros((len(veh.incomp),1))
			for lan in veh.incomp:
				for v_o in self.coord_set_[lan]: # use something like incomp_lane_list to reduce computations
					if v_o.exittime > curr_time:
						wait_time = round(max(wait_time, (v_o.exittime - curr_time), wait_time_from_prev_iter[v_o.lane]),1)
			functions.constraint_waiting_time(wait_time, self.X[j][0,:], opti)


			if (len(veh.incomp) != 0) and (j > 0):
				for k_i in range(len(veh.incomp)):
					opti.set_value(self.acces_var[j][k_i,:], large_pos_value)
				itr_list = list(range(j))
				itr_list.reverse()
				incomp_lane_list = np.zeros((len(veh.incomp),1))
				for k_o in itr_list:
					if (_sched[k_o].lane in veh.incomp) and (incomp_lane_list[veh.incomp.index(_sched[k_o].lane)] == 0):

						for k_c in range(self.steps + 1):
							opti.subject_to(self.X[j][0,:] <= cas.if_else((self.X[k_o][0,:] <= _sched[k_o].intsize + _sched[k_o].length), 0, large_pos_value))
							
						incomp_lane_list[veh.incomp.index(_sched[k_o].lane)] = 1
						if sum(incomp_lane_list) == len(veh.incomp):
						    break


		opti.minimize(-self.obj_fun)
		p = {"print_time": False, "ipopt.print_level": 0}
		opti.solver('ipopt', p)

		try:
			solve_init_time = time.time()
			temp_s = opti.solve()
			time_solve_duration = time.time() - solve_init_time
			self.solve_only_computation_time += time_solve_duration
			for j in range(len(_sched)):
				veh = _sched[j]

				self.pos[veh.lane].append(temp_s.value(self.X[j][0,:]))
				self.vel[veh.lane].append(temp_s.value(self.X[j][1,:]))
				self.acc_u[veh.lane].append(temp_s.value(self.acc[j][:]))
				self.each_veh_cost[veh.lane].append(temp_s.value(self.obj_fun_for_each_veh[j]))
				self.traf_eval[veh.lane].append(temp_s.value(self.traffic_eval_fun[j]))
			self.tot_cost = temp_s.value(self.obj_fun)

		except:
			if opti.return_status() == "Maximum_Iterations_Exceeded":
				print(f"Maximum_Iterations_Exceeded_{_sched_count}", opti.stats()["return_status"])
				self.max_iter_exceeded = True
							
			else:
				print(opti.return_status())
				self.max_iter_exceeded = True

		self.computation_time = time.time() - comp_init_time

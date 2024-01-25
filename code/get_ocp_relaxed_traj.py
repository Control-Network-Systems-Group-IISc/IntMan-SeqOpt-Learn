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


class ocp_relaxed_opt():
	def __init__(self, _prov_set, _coord_set, curr_time, _init_conds):

		opti = cas.Opti()

		self.disc_factor = 1

		self.max_iter_exceeded = False
		self.next_max_iter = 3000

		self.prov_set_ = copy.deepcopy(_prov_set)
		self.coord_set_ = copy.deepcopy(_coord_set)

		_sched = [_ for lane in self.prov_set_ for _ in lane]

		_sched_count = 0

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


		self.updated_prov_set = copy.deepcopy(self.prov_set_)

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
				self.obj_fun += veh.priority * (self.disc_factor ** i) * data_file.W_pos * self.X[j][1,i] * data_file.dt

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
				# print(a)

		for _l_ind, _l in enumerate(self.updated_prov_set):
			for j in range(len(_l)):
				self.updated_prov_set[_l_ind][j].p_traj = list(self.updated_prov_set[_l_ind][j].p_traj) + list(self.pos[_l_ind][j])
				self.updated_prov_set[_l_ind][j].v_traj = list(self.updated_prov_set[_l_ind][j].v_traj) + list(self.vel[_l_ind][j])
				self.updated_prov_set[_l_ind][j].u_traj = list(self.updated_prov_set[_l_ind][j].u_traj) + list(self.acc_u[_l_ind][j])

				for n in range(self.steps):
					self.updated_prov_set[_l_ind][j].t_ser.append(round((curr_time + (n*data_file.dt)), 1))

		self.computation_time = time.time() - comp_init_time
		


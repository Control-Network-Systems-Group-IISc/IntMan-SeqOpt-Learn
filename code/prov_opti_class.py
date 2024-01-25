# this file contains the class definition for optimization

import math
import casadi as cas
import csv
import numpy as np
import copy
import data_file
import functions

class prov_opti_class():
	def __init__(self, _veh, pre_v_, _t_init, _init_pos, _init_vel, _dura, _wait_time_for_fifo_dict):
		self.disc_factor = 1
		self._veh = copy.deepcopy(_veh)
		opti = cas.Opti()
		self.success = False

		self.steps = int(math.ceil(_dura/data_file.dt)) + 1

		self.X = opti.variable(2, self.steps+1)
		self.acc = opti.variable(1, self.steps+1)
		self.jerk = []

		for i in range(self.steps):
			self.jerk.append((self.acc[i+1] - self.acc[i]) / data_file.dt)
			
		self.obj_fun = 0

		for i in range(self.steps):
			self.obj_fun -= (data_file.W_acc * (self.acc[i]**2) * data_file.dt) + (data_file.W_jrk *(self.jerk[i]**2) * data_file.dt)
			self.obj_fun += (self.disc_factor ** i) * data_file.W_pos * self.X[1,i] * data_file.dt

		### construction of constraints ###

		### intersection safety ### 
		if data_file.used_heuristic == "fifo":
			_wait_time_for_fifo = 0
			for conflict_lane in _veh.incomp:
				_wait_time_for_fifo = max(_wait_time_for_fifo, _wait_time_for_fifo_dict[conflict_lane])

			functions.constraint_waiting_time(max(0, round(_wait_time_for_fifo - _t_init, 1)), self.X[0,:], opti)

		else:
			functions.prov_constraint_vel_max(self._veh, self.X[1,:], self.X[0,:], opti)

		### bounds ###
		functions.constraint_vel_bound(self._veh, self.X[1,:], opti)
		functions.constraint_acc_bound(self._veh, self.acc, opti)
		
		### initital conditions ###
		functions.constraint_init_pos(self.X, opti, _init_pos)
		functions.constraint_init_vel(self.X, opti, _init_vel)

		### discrete time dynamics ###
		
		for k in range(self.steps):
			x_next = functions.compute_state(self.X[:,k], self.acc[k], data_file.dt)
			functions.constraint_dynamics(self.X[:,k+1], x_next, opti)

		### rear end safety ###
		if pre_v_ != None:
			#print("preself._veh num_prov_phases:", pre_v_.num_prov_phases)
			ind = functions.find_index(pre_v_, _t_init)

			if self._veh.num_prov_phases > 0:
				if ( -round(self._veh.coord_init_pos - pre_v_.p_traj[ind], 4) >= round(pre_v_.length + max(0, -(((self._veh.coord_init_vel**2)/(2*self._veh.u_min)) - ((pre_v_.v_traj[ind]**2)/(2*pre_v_.u_min)))), 4) ):
					pass

				else:
					print("***********prov phase bad rear end", -round(self._veh.coord_init_pos - pre_v_.p_traj[ind], 4) + round(pre_v_.length + max(0, -(((self._veh.coord_init_vel**2)/(2*self._veh.u_min)) - ((pre_v_.v_traj[ind]**2)/(2*pre_v_.u_min)))), 4))
					print(a)

				if ( round(self._veh.coord_init_vel, 4) <= round(np.sqrt(2*self._veh.u_min*self._veh.coord_init_pos), 4) ):
					pass

				else:
					print("***********prov_phase bas intersection", round(np.sqrt(2*self._veh.u_min*self._veh.coord_init_pos), 4) - round(self._veh.coord_init_vel, 4) )
					print(a)
			functions.prov_constraint_rear_end_safety(self._veh, pre_v_, _t_init, self.X[0,:], self.X[1,:], opti) 

		
		opti.minimize(-self.obj_fun)

		
		p = {"print_time": False, "ipopt.print_level": 0} #, "ipopt.tol": 10**(-3), "ipopt.constr_viol_tol": 10**-3}#, "ipopt.bound_relax_factor": 0, "ipopt.honor_original_bounds": "yes"}#,"ipopt.bound_relax_factor": 0, "ipopt.honor_original_bounds": "yes", "ipopt.constr_viol_tol": 10**-3}#, "ipopt.bound_relax_factor":10**(-4)}#, "ipopt.tol": 10**-3}#, "ipopt.constr_viol_tol": 10**-3}#, "ipopt.acceptable_dual_inf_tol": 10**-3, "ipopt.acceptable_tol": 10**-3}#, "ipopt.acceptable_constr_viol_tol": 10**-3, "ipopt.acceptable_dual_inf_tol": 10**-3}# ,"ipopt.sb": "yes", "ipopt.constr_viol_tol": 10**-3, "ipopt.acceptable_tol": 10**-4}#, "ipopt.tol": 10**-4}
		opti.solver('ipopt', p)

		try:
			self.sol = opti.solve()
			self.pos = [opti.value(self.X[0,itera_ind]) for itera_ind in range(self.steps)]
			self.vel = [opti.value(self.X[1,itera_ind]) for itera_ind in range(self.steps)]
			self.acc = [opti.value(self.acc[itera_ind]) for itera_ind in range(self.steps)]

			for n in range(self.steps):
				self._veh.p_traj.append(self.pos[n])
				self._veh.v_traj.append(self.vel[n])
				self._veh.u_traj.append(self.acc[n])
				self._veh.t_ser.append(round((_t_init + (n*data_file.dt)), 1))

			for t in range(len(self._veh.t_ser)):
				self._veh.finptraj[self._veh.t_ser[t]] = self._veh.p_traj[t]
				self._veh.finvtraj[self._veh.t_ser[t]] = self._veh.v_traj[t]
				self._veh.finutraj[self._veh.t_ser[t]] = self._veh.u_traj[t]

		except Exception as e:
			self.pos = [opti.value(self.X[0,itera_ind]) for itera_ind in range(self.steps)]
			self.vel = [opti.value(self.X[1,itera_ind]) for itera_ind in range(self.steps)]
			self.acc = [opti.value(self.acc[itera_ind]) for itera_ind in range(self.steps)]

			for n in range(self.steps):
				self._veh.p_traj.append(self.pos[n])
				self._veh.v_traj.append(self.vel[n])
				self._veh.u_traj.append(self.acc[n])
				self._veh.t_ser.append(round((_t_init + (n*data_file.dt)), 1))

			for t in range(len(self._veh.t_ser)):
				self._veh.finptraj[self._veh.t_ser[t]] = self._veh.p_traj[t]
				self._veh.finvtraj[self._veh.t_ser[t]] = self._veh.v_traj[t]
				self._veh.finutraj[self._veh.t_ser[t]] = self._veh.u_traj[t]
	
			if pre_v_ != None:
				t_init_for_v = functions.find_index(pre_v_, self._veh.sp_t)
				for t_ind_for_v in range(len(self.pos)):
					if (t_init_for_v + t_ind_for_v >= len(pre_v_.p_traj)) or (ind == None) or (ind + t_ind_for_v >= len(pre_v_.p_traj)):
						break

					if round(pre_v_.p_traj[ind + t_ind_for_v] - self.pos[t_ind_for_v] - pre_v_.length, 3) < -0.005:

						print(f"\ncomputed difference: {pre_v_.p_traj[ind + t_ind_for_v] - self.pos [t_ind_for_v] - pre_v_.length}")
						if self._veh.num_prov_phases > 0:
							print(f"[computed]: current veh pos: {self.pos[t_ind_for_v]}\tprevself._veh_pos: {pre_v_.p_traj[ind + t_ind_for_v]}\ttime in prevself._veh: {pre_v_.t_ser[ind + t_ind_for_v]}\ttime in this veh: {round(self._veh.t_ser[-1] + data_file.dt*(t_ind_for_v+1), 1)}\tprevself._veh length: {pre_v_.length}")

						else:
							print(f"[computed]: current veh pos: {self.pos[t_ind_for_v]}\tprevself._veh_pos: {pre_v_.p_traj[ind + t_ind_for_v]}\ttime in prevself._veh: {pre_v_.t_ser[ind + t_ind_for_v]}\ttime in this veh: {round(self._veh.sp_t + data_file.dt*(t_ind_for_v+1), 1)}\tprevself._veh length: {pre_v_.length}")
						

						print("ERROR!!!!!")
						print("veh p_traj:", self._veh.p_traj)
						print("veh v_traj:", self._veh.v_traj)
						print("veh _init_pos:", _init_pos)
						print("veh _init_vel:", _init_vel)
						print("veh t_ser:", self._veh.t_ser)
						print("veh sp_t:", self._veh.sp_t)
						try:
							print("veh last time:", self._veh.t_ser[-1])
						except:
							pass
						print("veh num_prov_phases:", self._veh.num_prov_phases)

						print("veh coord_init_pos:", self._veh.coord_init_pos)
						print("veh coord_init_vel:", self._veh.coord_init_vel)

						with open(f"../data/infeasiblity_record.csv", "a", newline="") as f:
							writer = csv.writer(f)
							writer.writerows([[_t_init, "prov_phase"]])
							writer.writerows([[*self._veh.p_traj, _init_pos]])
							writer.writerows([[*self._veh.v_traj, _init_vel]])
							writer.writerows([self._veh.u_traj])
							writer.writerows([[*self._veh.t_ser, _t_init]])

							if not (pre_v_ == None):
								writer.writerows([pre_v_.p_traj])
								writer.writerows([pre_v_.v_traj])
								writer.writerows([pre_v_.u_traj])
								writer.writerows([pre_v_.t_ser])

							writer.writerows([["-","-","-","-","-","-","-","-","-","-","-","-"]])
						# print(a)

			return
			
			# print(a)

# this file contains the function prov_phase().
# it takes in a vehicle object, previous vehicle on that lane, current time
# and webster estimate of simulation time as arguments.
# it over-writes the vehicle object with updated trajectory of the vehicle 
# till the end of its provisional phase

import matplotlib.pyplot as plt
import prov_opti_class
import data_file
import functions
 

def prov_phase(veh, _pre_v, t_init, wait_time_for_fifo_dict):

	init_pos0 = veh.p0
	init_vel0 = veh.v0

	if len(veh.t_ser) <= 1:
		veh.t_ser = []
		veh.p_traj = []
		veh.v_traj = []

	elif len(veh.p_traj) > 1:
		temp_ind = functions.find_index(veh, t_init)
		init_pos0 = veh.p_traj[temp_ind]
		init_vel0 = veh.v_traj[temp_ind]

		veh.p_traj = veh.p_traj[:temp_ind]
		veh.v_traj = veh.v_traj[:temp_ind]
		veh.u_traj = veh.u_traj[:temp_ind]
		veh.t_ser = veh.t_ser[:temp_ind]
		
	else:
		pass



	if _pre_v != None:
		index = functions.find_index(_pre_v, t_init)

	else:
		pass


	if data_file.used_heuristic == "fifo":
		dura = data_file.T_sc

	else:
		if veh.num_prov_phases == 0:
			dura = round((data_file.t_opti - (t_init%data_file.t_opti) + ((data_file.prov_extra_steps)*data_file.dt)), 1)		
		else:
			dura = round((data_file.t_opti  + ((data_file.prov_extra_steps)*data_file.dt)), 1)

	ps = None
	ps = prov_opti_class.prov_opti_class(veh, _pre_v, t_init, init_pos0, init_vel0, dura, wait_time_for_fifo_dict)

	if data_file.used_heuristic == "fifo":
		if False:
			while ps._veh.p_traj[-2] < (ps._veh.intsize + ps._veh.length):
				first_pos = ps._veh.p_traj[-1] + (ps._veh.v_traj[-1]*data_file.dt) + (0.5*(data_file.dt**2)*ps._veh.u_traj[-1])
				first_vel = ps._veh.v_traj[-1] + (ps._veh.u_traj[-1]*data_file.dt)
				ps = prov_opti_class.prov_opti_class(ps._veh, _pre_v, round(ps._veh.t_ser[-1]+data_file.dt, 1), first_pos, first_vel, dura, wait_time_for_fifo_dict)

		if True:
			temp_num_iter_fifo = 2
			while ps._veh.p_traj[-2] < (ps._veh.intsize + ps._veh.length):
				ps = prov_opti_class.prov_opti_class(veh, _pre_v, t_init, init_pos0, init_vel0, temp_num_iter_fifo*dura, wait_time_for_fifo_dict)
				temp_num_iter_fifo += 1

	ps._veh.num_prov_phases += 1

	return ps._veh, True

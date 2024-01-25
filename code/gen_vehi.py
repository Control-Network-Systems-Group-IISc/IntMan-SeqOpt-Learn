# this file contains the function definition for generating vehicles
# it takes number of vehicles per lane and arrival rate as arguments
# it returns an array of vehicle objects with those many vehicles in a each lane
# with the specified arrival rate and the webster estimate for simulation time.


##### NOTE: THESE VEHICLE OBJECTS MAY NOT SATISFY THE INITIALIZATION CONSTRAINTS ######

import numpy as np
import data_file
import vehicle
from collections import deque


def gen_vehi(veh_per_lane_array, arr_rate_array):
	v_arr = []
	tclr = 0
	generated_veh = [deque([]) for _ in data_file.lanes]
	id_var = 0
	for b in range(len(data_file.lanes)):
		arr =  arr_rate_array[b]
		array = []
		for i in range(veh_per_lane_array[b]):
			v = vehicle.Vehicle(data_file.lanes[b], data_file.int_start[data_file.lanes[b]], 0, data_file.vm[data_file.lanes[b]], data_file.u_min[data_file.lanes[b]], data_file.u_max[data_file.lanes[b]], data_file.L, arr_rate_array)
			v.lane = data_file.lanes[b]
			v.incomp = data_file.incompdict[v.lane]
			v.intsize = data_file.intersection_path_length[v.lane%3]
			v.arr = arr
			if i == 0:
				v.sp_t = round(float(np.random.poisson(1/arr,1)),1)
				v.id = id_var
				array.append(v)

			else:
				v.sp_t = round(array[-1].sp_t + float(np.random.exponential(1/arr,1)),1)
				v.id = id_var
				array.append(v)

			id_var += 1

		generated_veh[b] = deque(array)

	return generated_veh, tclr

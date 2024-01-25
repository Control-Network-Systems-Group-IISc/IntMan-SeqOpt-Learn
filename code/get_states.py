# this file defines the function to get states/features of the vehicles in
# the set _V_set

import copy

import data_file
import functions


### !!!!!!!!!! CAUTION !!!!!!!!! ###

######## DO NOT CHANGE THE ORDER OF APPENDING THE FEATURES/STATE VARIABLES #######

def get_states(_t_inst, _Vp_set, _coord_set_):

	state = [] 

	for li in range(len(data_file.lanes)):
		l = data_file.lanes[li]
		for ind in range(len(_Vp_set[l])):

			v = _Vp_set[l][ind]

			t_v = functions.find_index(v, round(_t_inst, 1))

			if t_v != None:
				assert not (((t_v == -1) and (v.p_traj[t_v] < data_file.p_init)))
				pos = v.p_traj[t_v]
				vel = v.v_traj[t_v]
			elif t_v == None:
				pos = v.p0
				vel = v.v0

			v.feat_d_s_arr = -v.int_start + pos
			state.append(v.feat_d_s_arr)

			v.feat_v = vel
			state.append(v.feat_v)

			if len(v.t_ser) > 0:
				v.feat_t_s_arr = _t_inst - v.t_ser[0]
			else:
				v.feat_t_s_arr = 0
			state.append(v.feat_t_s_arr)

			no_v_f = 0
			v_fol = []

			for v_i in list(_Vp_set[v.lane])[ind+1:]:
				no_v_f += 1
				v_fol.append(v_i)

			v.feat_no_v_follow = no_v_f
			state.append(v.feat_no_v_follow)

			avg_seperation = v.feat_d_s_arr
			if no_v_f > 0:
				avg_seperation = 0
				for v_f in v_fol:
					t_v_f = functions.find_index(v_f, _t_inst)
					if t_v_f != None:
						avg_seperation += pos - v_f.p_traj[t_v_f] # v.p_traj[t_v] - v_f.p_traj[t_v_f]

					else:
						avg_seperation += pos - v_f.p0

				avg_seperation = avg_seperation/(no_v_f)

			v.feat_avg_sep = avg_seperation
			state.append(v.feat_avg_sep)

			# v.feat_avg_arr_rate = v.arr
			state.append(v.feat_avg_arr_rate)

			wait_time = 0

			for ln in v.incomp:
				for v_o in _coord_set_[ln]:
					wait_time = max(wait_time, (v_o.exittime - _t_inst))

			v.feat_min_wait_time = round(wait_time,1)

			state.append(v.feat_min_wait_time)

			state.append(v.lane)
			_Vp_set[l][ind] = copy.deepcopy(v)

			state.append(v.v_max)
			state.append(v.u_max)

			state.append(v.priority)
		
	return state, _Vp_set

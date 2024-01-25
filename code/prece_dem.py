# this file contains code to claculate precedence indices and demand for DDSWA

import numpy as np
import copy
import itertools
import data_file
import functions
import get_states

import tensorflow as tf

###!!!!!!!!! CAUTION !!!!!!!!!###
## DO NOT CHANGE THE ORDER OF APPENDING THE piS AND diS IN ANY WAY ##

def sigmoid(x):
	return 1/(1+np.exp(-x))
	
def get_prece_dem_RL(t_inst, _prov_set, _coord_set, rl_agent, alg_option, l_flag):

	exlporation_flag = 0

	rl_state = []

	rl_action = []

	p_index = []

	dem = []

	Vp_set = _prov_set

	if data_file.rl_algo_opt == "MADDPG":
		flatten = itertools.chain.from_iterable
	
	if alg_option == "rl_modified_ddswa":
		Vp_set = _prov_set
		

	elif alg_option == "rl_ddswa":
		Vp_set = functions.get_set_f(_prov_set)

	V_states, _Vp_set = get_states.get_states(t_inst, Vp_set, _coord_set)


	if alg_option == "rl_modified_ddswa":

		if data_file.rl_algo_opt == "DDPG":

			if alg_option == "rl_ddswa":

				for i in range((data_file.lane_max) - functions.get_num_of_objects(Vp_set)):
					V_states.append(data_file.d_since_arr)
					V_states.append(data_file.feat_vel)
					V_states.append(data_file.t_since_arr)
					V_states.append(data_file.no_v_follow)
					V_states.append(data_file.avg_sep)
					V_states.append(data_file.avg_arr_rate)
					V_states.append(data_file.min_wait_time)
					V_states.append(data_file.lane)

				rl_state = np.asarray(V_states).reshape(1, (data_file.num_features*data_file.lane_max))
			
				rand = np.random.uniform(0,1)


				if rand > 0.05:
					rl_action = (act_nn_object._actor._model.predict(rl_state)[0] + 1).reshape((2*data_file.lane_max), 1)
				
				else:
					exlporation_flag = 1
					rl_action = (act_nn_object._actor._model.predict(rl_state) + np.random.uniform(low=1, high=25, size=(2*len(data_file.lanes)))).reshape((2*data_file.lane_max), 1)
				
				p_index = []
				dem = []

				for i in range(functions.get_num_of_objects(Vp_set)):
					p_index.append(rl_action[i])
					dem.append(rl_action[(data_file.lane_max)+i])


			if alg_option == "rl_modified_ddswa":
				for i in range((data_file.num_veh) - functions.get_num_of_objects(Vp_set)):
					V_states.append(data_file.d_since_arr)
					V_states.append(data_file.feat_vel)
					V_states.append(data_file.t_since_arr)
					V_states.append(data_file.no_v_follow)
					V_states.append(data_file.avg_sep)
					# V_states.append(data_file.avg_arr_rate)
					V_states.append(data_file.min_wait_time)
					V_states.append(data_file.lane)
					V_states.append(data_file.vel_bound)
					V_states.append(data_file.acc_bound)
					V_states.append(data_file.priority_)

				rl_state = np.asarray(V_states[0:data_file.num_features*data_file.num_veh]).reshape(1, data_file.num_features*data_file.num_veh)

				if l_flag:
					rl_action = rl_agent.policy(rl_state, rl_agent.ou_noise)[0]

				else:
					rl_action = tf.squeeze(rl_agent.actor_model_(rl_state)).numpy()
					
				p_index = []
				dem = []

				for i in range(functions.get_num_of_objects(Vp_set)):
					p_index.append(rl_action[i])
					dem.append(1)
				

		
		if alg_option == "rl_modified_ddswa":
			lane_itr_var = 0
			for li in range(len(data_file.lanes)):
				l = data_file.lanes[li]
				if len(_prov_set[l]) > 0:
					for ind in range(len(_prov_set[l])):
						_prov_set[l][ind].priority_index = p_index[lane_itr_var]
						_prov_set[l][ind].demand = dem[lane_itr_var]
						lane_itr_var += 1


	else:

		if alg_option == 'ocp':
			temp_p_set = copy.deepcopy(_prov_set)
			temp_c_set = copy.deepcopy(_coord_set)

			temp_pi_updated_p_set = functions.get_ocp_prece_indices(temp_p_set, temp_c_set, t_inst)


		for li in range(len(data_file.lanes)):
			l = data_file.lanes[li]
			if len(_prov_set[l]) > 0:
				for ind in range(len(_prov_set[l])):

					if alg_option == 'ocp':
						p_index.append(temp_pi_updated_p_set[l][ind].priority_index)

					elif alg_option == "congestion":
						p_index.append(- _prov_set[l][ind].feat_avg_sep)

					elif alg_option == "num_follow_veh":
						p_index.append(_prov_set[l][ind].feat_no_v_follow)

					elif alg_option == "dist_to_int":
						p_index.append(-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr)

					elif alg_option == "norm_dist_and_vel":
						p_index.append((- (_prov_set[l][ind].feat_d_s_arr / _prov_set[l][ind].p0)) + (_prov_set[l][ind].feat_v / _prov_set[l][ind].v_max) )

					elif alg_option == "time_to_react":

						if _prov_set[l][ind].feat_v > 0:
							p_index.append( -(-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr)/(_prov_set[l][ind].feat_v) )

						else:
							p_index.append( -(-_prov_set[l][ind].p0)/(10**(-12)) )

					elif alg_option == "dist_react_time":

						if _prov_set[l][ind].feat_v > 0:
							p_index.append(-((-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr)** 2) /(_prov_set[l][ind].feat_v) )

						else:
							p_index.append(-((-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr)**2)/(10**(-12)) )


					elif alg_option == "conv_dist_react":

						if _prov_set[l][ind].feat_v > 0:
							p_index.append(-((0.5* (-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr)) + (0.5 * ((-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr)) /(_prov_set[l][ind].feat_v))) )

						else:
							p_index.append(-((0.5* (-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr)) + (0.5 * ((-_prov_set[l][ind].p0 - _prov_set[l][ind].feat_d_s_arr))/(10**(-12)))) )


					elif alg_option == "fifo":

						p_index.append( -(prov_set[l][ind].sp_t) )


					dummy_dem = []
					for j in range(1, data_file.num_dem_param+1):
						if j == 1:
							dummy_dem.append(5)

						elif j == 2:
							dummy_dem.append(0)

						elif j == 3:
							dummy_dem.append(0)

						elif j == 4:
							dummy_dem.append(0)

						elif j == 5:
							dummy_dem.append(0)

					dem.append(dummy_dem)



		lane_itr_var = 0
		for li in range(len(data_file.lanes)):
			l = data_file.lanes[li]
			if len(_prov_set[l]) > 0:
				for ind in range(len(_prov_set[l])):
					if p_index[lane_itr_var] == None:
						print(a)
					_prov_set[l][ind].priority_index = p_index[lane_itr_var]
					_prov_set[l][ind].demand = 1 # dem[lane_itr_var]
					lane_itr_var += 1

	return _prov_set, p_index, dem, rl_state, rl_action, exlporation_flag
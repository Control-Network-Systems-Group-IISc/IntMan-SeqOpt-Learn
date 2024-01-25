import pickle
import data_file
import gen_vehi
import set_class


for arr_rate in data_file.arr_rates_to_simulate:

	for sim in range(1, data_file.num_sim + 1):
		
		sim_obj = set_class.sets()

		arr_rate_dict = arr_rate_array = {0:0, 1:arr_rate, 2:arr_rate, 3:0, 4:arr_rate, 5:arr_rate, 6:0, 7:arr_rate, 8:arr_rate, 9:0, 10:arr_rate, 11:arr_rate}

		sim_obj.unspawned_veh, webstr_time = gen_vehi.gen_vehi(data_file.no_veh_per_lane, arr_rate_dict)

		dbfile = open(f'../data/compare_files/homogeneous_traffic/arr_{arr_rate}/sim_obj_num_{sim}', 'wb')
		pickle.dump(sim_obj, dbfile)
		dbfile.close()
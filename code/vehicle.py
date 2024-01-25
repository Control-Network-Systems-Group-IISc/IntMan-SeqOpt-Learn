# This file defines the Vehicle object

import data_file
import numpy as np

class Vehicle():
      
      def __init__(self, lane_number, init_pos, v_min, v_max, u_min, u_max, length, arr_rate_array):
            self.arr = arr_rate_array[lane_number]
            self.sp_t = None # spawning time to be initialized
            
            self.lane = data_file.lanes[lane_number]
            self.int_start = data_file.int_start[self.lane]
            self.length = length
            self.u_max = u_max
            self.u_min = u_min
            self.v_min = v_min
            self.v_max = v_max

            self.p0 = round(init_pos,1)
            self.v0 = np.random.uniform(self.v_min, self.v_max)
            self.p_traj = []
            self.v_traj = []
            self.u_traj = []
            self.t_ser = []
            self.ptime = None # provisional phase time to be initialized
            self.stime = None # coordinated phase time
            self.id = None
            self.finptraj = {}
            self.finvtraj = {}
            self.finutraj = {}
            self.exittime = None
            self.priority_index = None
            self.demand = None
            self.priority = np.random.choice([1, 2, 4, 5], p=[0.5, 0.3, 0.15, 0.05]) #data_file.lane_priorities[self.lane]
            self.incomp = data_file.incompdict[self.lane]
            self.intsize = data_file.intersection_path_length[lane_number%3]
            self.coord_cpcost = 0
            self.comb_opt_like_cost = 0
            self.traffic_eval_func_val = 0
            self.re_prov_flag = 0

            self.coord_init_pos = None
            self.coord_init_vel = None
            self.num_prov_phases = 0

            #features of vehicle
            self.feat_d_s_arr = data_file.d_since_arr
            self.feat_v = data_file.feat_vel
            self.feat_t_s_arr = data_file.t_since_arr
            self.feat_no_v_follow = data_file.no_v_follow
            self.feat_avg_sep = data_file.avg_sep
            self.feat_avg_arr_rate = data_file.avg_arr_rate
            self.feat_min_wait_time = data_file.min_wait_time
            self.feat_lane = data_file.lane
# This file has all the data/variables required in one place

import numpy as np

####### RL or otherwise? #######
# set rl_flag to 1 if RL and 0 otherwise
rl_flag = 1

rl_algo_opt = "DDPG" # available options: DDPG or MADDPG
####### RL or otherwise? #######

algo_option = "rl_modified_ddswa"

# simulation time limit
max_sim_time = 500

real_time_spawning_flag = 1

run_coord_on_captured_snap = 0

heuristic_dict_id = 0

heuristic_dict = {0: None, 1:"fifo", 2:"time_to_react", 3:"dist_react_time", 4:"conv_dist_react", 5:"ocp"}

used_heuristic = heuristic_dict[heuristic_dict_id]

# arr_rates_to_simulate = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
# arr_rates_to_simulate = [0.125, 0.175, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
arr_rates_to_simulate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# arr_rates_to_simulate = [0.2]


# time step
dt = round(1.0, 1)

# weight in cost on velocity
W_pos = 1

# weight in cost on acceleration
W_acc = 0

# weight in cost on jerk
W_jrk = 0

# scaling factor for average demand in ddswa
w_l = 1

# upper-bound on velocity
vm = {0:1, 3:1, 6:1, 9:1, 1:1.5 ,2:1 ,4:1, 5:1.5, 7:1.5, 8:1, 10:1, 11:1.5}
#{0:1.5, 3:1.5, 6:1.5, 9:1.5, 1:1.5 ,2:1.5 ,4:1.5, 5:1.5, 7:1.5, 8:1.5, 10:1.5, 11:1.5}

# upper-bound on acceleration
u_max = {0:2, 3:2, 6:2, 9:2, 1:2, 2:2, 4:2, 5:2, 7:2, 8:2, 10:2, 11:2} #3.0

# lower-bound on acceleration
u_min = {0:-2, 3:-2, 6:-2, 9:-2, 1:-2, 2:-2, 4:-2, 5:-2, 7:-2, 8:-2, 10:-2, 11:-2} #-3.0

# maximum number of vehicles in a lane related to RL
max_vehi_per_lane = 15

# RL DDPG replay-buffer-size
rl_ddpg_buff_size = 256

# RL DDPG sample-size !!!! important: keep this an even number !!!!
rl_ddpg_samp_size = 64

# RL Rpioritized experience replay (PER) flag. 0 means no PER
rl_per_flag = 0

# lane numbers
lanes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# maximum number of lanes
lane_max = len(lanes)

# number of lanes per direction
lane_per_dir = lane_max/4

# Length of a vehicle
act_len = 0.75

buffer = 0 * act_len

L = act_len + (2*buffer)

# Width of vehicle
B = 0.7

# Number of lanes per branch
num_lanes = 2

num_veh = 10 * 4 * num_lanes

# length of the intersection (actual intersection)
int_bound = 2 * num_lanes * B

#lanes = [1, 2, 4]

# start of region of interest on different lanes
int_start = {0:-7, 3:-7, 6:-7, 9:-7, 1:-7 ,2:-7, 4:-7, 5:-7, 7:-7, 8:-7, 10:-7, 11:-7}

# distance between lanes
dist_bw_lanes = int_bound/(2*lane_per_dir)

# scheduling time
T_sc = 60

# reward calculation time
if T_sc == 30:
	T_r = 20
if T_sc == 60:
	T_r = 30

# optimization interval T_c
t_opti = 6

# number of additional time steps to plan for after end of provisional phase 
# to maintain trajectory continuity in coordinated phase
prov_extra_steps = 2 # keep this value greater than or equal to 2


# lenght of path inside intersection
intersection_path_length = [B / np.sqrt(2), int_bound, np.sqrt(((2.5 * B) ** 2) + ((2.5 * B) ** 2))]


# incompatibility dictionary
incompdict = {0:[],3:[],6:[],9:[],1:[4,10,8,11],2:[4,7,5,11],4:[1,7,2,11],5:[7,10,2,8],7:[4,10,2,5],8:[1,10,5,11],10:[1,7,8,5],11:[4,1,8,2]}


# initial spawning position
p_init = int_start

#### data for main.py file ####
num_sim = 100 # number of simulations to be done
veh_in_lane = 1000
no_veh_per_lane = {0:0, 1:veh_in_lane, 2:veh_in_lane, 3:0, 4:veh_in_lane, 5:veh_in_lane, 6:0, 7:veh_in_lane, 8:veh_in_lane, 9:0, 10:veh_in_lane, 11:veh_in_lane} # number of vehicles per lane to be considered in each simulation

#### features of pseudo vehicle ####
#!!!!!!!! CAUTION !!!!!!!!#
######## DO NOT CHANGE THE ORDER OF THE FEATURES/STATE VARIABLES #######
d_since_arr = -7
feat_vel = 0
t_since_arr = -6
no_v_follow = 0
avg_sep = -10
avg_arr_rate = 0
min_wait_time = 60
lane = -1
vel_bound = 0
acc_bound = -10
priority_ = -10

# number of fetures
num_features = 10 #no arr_rate feature
num_dem_param = 5


#current_used_weights = weights_Wc0_Wv1
colours = {0:'black', 1:'blue', 2:'brown', 3:'cyan', 4:'green', 5:'navy', 6:'grey', 7:'violet', 8:'olive', 9:'magenta', 10:'red', 11:'fuchsia'}





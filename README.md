# Reinfrocement Learning Aided Sequential Optimization for Autonomous Intersection Management

## Description
Welcome to our project! In this repository, we have evaluated the autonomous intersection managemnt method proposed in our paper named "Reinforcement Learning Aided Sequential Optimization for Unsignalized Intersection Management of Robot Traffic" (pre-print available at https://arxiv.org/abs/2302.05082). The proposed solution framework gives safe and efficient trajectories for streams of robots arriving randomly at an unsignalized isolated intersection. A Deep deterministic policy gradient agent is used for policy learning deciding the robot crossing order.

Currently, the code includes the following techniques to get trajectories. 
  - Two exaustive search based optimization techniques
      - Combined optimization (non-linear optimzation program)
      - Best sequence optimization (smaller convex programs solved sequentially) 
  - Specific sequence based optimization framework where the sequence is given by
      - Collect-Merge-Learn (CML) based trained RL agent (including the code to train the agent)
      - first come first serve rule (FCFS)
      - ratio of current distance to intersection and velocity (time to react or TTR -- G. R. de Campos, P. Falcone et al. in 2013)
      - convex combination of time to react and distance to intersection (CDT -- N. Suryarachi, F. M. Tariq et al. in 2021)
      - product of time to react and distance to intersection (PDT -- N. Suryarachi, R. Chandra et al.  in 2020)
      - solving an optimization problem to get the crossing order (OCP -- X. Pan, B. Chen et al. in 2023)
      

## Instructions
1. Update the arrival rates, simulation duration, velocity and acceleration limits etc, in data_file.py
2. Run the shell script
   - run_tr-te.sh to train and evaluate the CML based RL policies
   - run_heuristics to run the other heuristics (FCFS, TTR, CDT, PDT, OCP)



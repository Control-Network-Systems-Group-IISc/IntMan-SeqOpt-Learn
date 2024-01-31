# Reinfrocement Learning Aided Sequential Optimization for Autonomous Intersection Management

## Description
Welcome to our project! In this repository, we have evaluated the autonomous intersection managemnt method proposed in our paper named "Reinforcement Learning Aided Sequential Optimization for Unsignalized Intersection Management of Robot Traffic" (pre-print available at https://arxiv.org/abs/2302.05082). The proposed solution framework gives safe and efficient trajectories for streams of robots arriving randomly at an unsignalized isolated intersection.

Currently, the code includes the following techniques to get trajectories. 
  - Two exaustive search based optimization techniques
      a. Combined optimization (non-linear optimzation program)
      b. Best sequence optimization (smaller convex programs solved sequentially) 
  - Specific sequence based optimization framework where the sequence is given by
      a. Collect-Merge-Learn based trained RL agent (including the code to train the agent)
      b. first come first serve rule
      c. ratio of current distance to intersection and velocity (time to react -- G. R. de Campos, P. Falcone et al. in 2013)
      d. convex combination of time to react and distance to intersection (CDT -- N. Suryarachi, F. M. Tariq et al. in 2021)
      e. product of time to react and distance to intersection (PDT -- N. Suryarachi, R. Chandra et al.  in 2020)
      f. solving an optimization problem to get the crossing order (OCP -- X. Pan, B. Chen et al. in 2023)
      

A Deep deterministic policy gradient agent is used for policy learning deciding the robot crossing order

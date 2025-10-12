
import pandas as pd
import numpy as np
import json
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model.AGVVRP import AGVVRPOPTIM

how='ga'
dist_matrix=pd.read_pickle('E:/glass_git/AGV_VRP/data/distance_matrix.pkl').to_numpy()
task=pd.read_csv('E:/glass_git/AGV_VRP/data/task_re.csv').to_numpy()
agv=pd.read_csv('E:/glass_git/AGV_VRP/data/agv_re.csv').to_numpy()

save_path='E:/glass_git/AGV_VRP/result/solution'
population=100
global_search_iteration=100000
local_search_iteration=30
init_params={'time_dist':0.4,'dist':0.4,'random':0.2, 'pox':0.4, 'jbx':0.4, 'mutation':0.2, 'global_drop_rate':0.2}

agvvrp=AGVVRPOPTIM(agv,task,dist_matrix,population,global_search_iteration,local_search_iteration,init_params,save_path)
best_solution=agvvrp.run()
print(best_solution)

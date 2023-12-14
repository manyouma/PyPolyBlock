import numpy as np
from numpy import linalg as LA
import vertexset 


param = {
         "tor_BS": 0.001,  # Accuracy for bisection search
         "tor_MO": 0.01,   # Accuracy for monotonic optimzation 
         "tor_AXIS": 0.001, # Torlerance for removing the vertices that are too closed to the axis
         "Dim": 8,          # Number of varialbes 
         "init_point": 1.5, # Starting point of monotonic optimization
         "freq_improper": 1000, 
         "freq_dominated": 100,
         "freq_print":1,
        }

# Objective Function to Maximize
def objective(x):
    return np.sum(np.power(x,1))

# Funtion that defines the feasibility reigion
def feasibility(x):
    if np.sum(np.power(x,2)) <= 10.:
        return True
    else: 
        return False

myset = vertexset(param, param["init_point"]*np.ones(param["Dim"]), objective, feasibility)
myset.refine_index(0)

num_iteration = 0
while np.max(myset.value_vec) >= myset.best_value + param["tor_MO"]:
    
    # remove vertices that are too closed to the axis
    myset.remove_closed2axis()
    
    # Pick the vertex with the current upper bound (CUB)
    chosen_index = np.argmax(myset.value_vec)
    myset.refine_index(chosen_index) 

    if np.mod(num_iteration, param["freq_print"]) == 1:
        print(f"ITER:{num_iteration}, CBV: {myset.best_value}, CUB: {np.max(myset.value_vec)}, chosen_index: {chosen_index}, num_vertex: {len(myset.value_vec)}")
        
    if np.mod(num_iteration, param["freq_improper"]) == 1:
        myset.remove_improper()

    if np.mod(num_iteration, param["freq_dominated"]) == 1:
        myset.remove_dominated()

    num_iteration += 1
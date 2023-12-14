import numpy as np
from numpy import linalg as LA
from vertexset import vertexset


param = {
         "tor_BS": 0.001,   # Accuracy for bisection search
         "tor_MO": 0.01,    # Accuracy for monotonic optimzation 
         "tor_AXIS": 0.001, # Torlerance for removing the vertices that are too closed to the axis
         "Dim": 3,          # Number of varialbes 
         "init_point": 3, # Starting point of monotonic optimization
         "freq_improper": 1000, 
         "freq_dominated": 100,
         "freq_print":100,
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

V = vertexset(param, param["init_point"]*np.ones(param["Dim"]), objective, feasibility)
V.refine_index(0)

num_iteration = 0
while np.max(V.value_vec) >= V.best_value + param["tor_MO"]:
    
    # remove vertices that are too closed to the axis
    V.remove_closed2axis()
    
    # Pick the vertex with the current upper bound (CUB)
    chosen_index = np.argmax(V.value_vec)
    V.refine_index(chosen_index) 

    if np.mod(num_iteration, param["freq_print"]) == 0:
        print(f"ITER:{num_iteration}, CBV: {V.best_value}, CUB: {np.max(V.value_vec)}, chosen_index: {chosen_index}, num_vertex: {len(V.value_vec)}")
        
    if np.mod(num_iteration, param["freq_improper"]) == 1:
        V.remove_improper()

    if np.mod(num_iteration, param["freq_dominated"]) == 1:
        V.remove_dominated()

    num_iteration += 1
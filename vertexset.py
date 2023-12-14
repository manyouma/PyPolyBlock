import numpy as np
from numpy import linalg as LA

class vertexset:
    def __init__(self, param, init_vertex, obj_func, feas_func):
        self.param = param
        self.dim = param["Dim"]
        self.objective = obj_func
        self.feasibility = feas_func

        # These two fields should always be updated together
        self.value_vec = np.array([self.objective(init_vertex)])
        self.set_vec =  np.zeros((1, self.dim))
        
        # Initialize the algorithm 
        self.set_vec[0,:] = init_vertex
        self.best_value = 0 
        self.best_vertex = []

    def remove_improper(self):
        removed = np.array([], dtype=int)
        num_vertex = len(self.value_vec)
        
        for i in np.arange(num_vertex):
            if np.sum(np.sum(self.set_vec > self.set_vec[i,:], axis=1)==self.dim) >= 1:
                removed = np.concatenate((removed, np.array([i])))
        
        if removed.shape[0] >= 1:
            self._remove_vertex(removed.astype(int))
            print(f"Removed {removed.shape[0]} improper vertices.")
        return


    def remove_dominated(self):
        if len(self.value_vec) > 0 and np.sum(self.best_value+self.param["tor_MO"] > self.value_vec) >= 1:
            self._remove_vertex(np.where(self.best_value > self.value_vec))
        return 

    def remove_closed2axis(self):
        removed = np.where(np.sum(self.set_vec < self.param["tor_AXIS"], axis=1))
        if len(removed) >= 1:
            self._remove_vertex(removed)
        return
        
    def refine_index(self, index):
        new_max, new_min = self._bisection_vertex(index)
        if self.objective(new_min) > self.best_value:
            self.best_value = max(self.objective(new_min), self.best_value)
            self.best_vertex = new_min
        add_index = self._split_vertex(index, new_max)
        # print(add_index)
        
        self._add_vertex(add_index)
        self._remove_vertex(index)
    
    
    def _split_vertex(self, index, x_new):
        x_ori = self.set_vec[index,:]
        output = np.zeros((self.dim, self.dim))
        for i_output in np.arange(self.dim):
            output[i_output, :] = x_ori
            output[i_output, i_output] = x_new[i_output]

        return output   
    
    def _bisection_vertex(self, index):
        current_max = self.set_vec[index,:]
        current_min = np.zeros_like(current_max)
        while LA.norm(current_max - current_min) > self.param["tor_BS"]:
            try_x = (current_max + current_min)/2
            if self.feasibility(try_x):
                current_min = try_x
            else:
                current_max = try_x
        return current_max,current_min
        
    def _add_vertex(self, new_vertex):
        num_add, dim = new_vertex.shape
        value_new = np.zeros(num_add)
        
        for i in np.arange(num_add):
            value_new[i] = self.objective(new_vertex[i,:])
        
        self.value_vec = np.concatenate((self.value_vec, value_new))
        self.set_vec = np.vstack((self.set_vec, new_vertex))
        return

    def _remove_vertex(self, index):
        self.value_vec = np.delete(self.value_vec, index, axis=0)
        self.set_vec = np.delete(self.set_vec, index, axis=0)
        return
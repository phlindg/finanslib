'''
Created on 18 juni 2017

@author: Phili
'''

import numpy as np
from mylib.sndRNG import sn_random_numbers
from mylib.simulation_class import simulation_class

class sqrt_diffusion(simulation_class):
    
    def __init__(self, name, mar_env, corr=False):
        super(sqrt_diffusion, self).__init__(name, mar_env, corr)
        try:
            self.kappa = mar_env.get_constant("kappa")
            self.theta = mar_env.get_constant("theta")
        except:
            print("Error getting stuff for env")
    
    def update(self, initial_value = None, volatility = None, kappa=None, theta = None,final_date = None):
        if initial_value != None:
           self.initial_value = initial_value
        if volatility != None:
            self.volatility = volatility
        if kappa != None:
            self.kappa = kappa
        if theta != None:
            self.theta = theta
        if final_date != None:
            self.final_date = final_date
        self.instrument_values = None #VARFOR
        
    def generate_paths(self, fixed_seed = False, day_count = 365.):
        if self.time_grid == None:
            self.generate_time_grid()
        #number of dates in time grid
        M = len(self.time_grid)
        #number of paths
        I = self.paths
        #array initalization for path simulation
        paths = np.zeros((M,I))
        paths_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        paths_[0] = self.initial_value
        if not self.correlated:
            #If not correlated, generate random numbers
            rand = sn_random_numbers((1, M, I),fixed_seed = fixed_seed)
        else:
            #if not correlated, use random numbers in market env
            rand = self.random_numbers
        #get short rate for drift of processes
        short_rate = self.discount_curve.short_rate
        for t in range(1,len(self.time_grid)):
            #select the right time slice from the relevant time sets
            if not self.correlated:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:,t,:])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t-1]).days / day_count
            #diff between two dates as year fraction
            paths_[t] = (paths_[t-1] + self.kappa * 
                         (self.theta-np.maximum(0,paths_[t-1, :]))*dt
                         +np.sqrt(np.maximum(0, paths_[t-1, :]))
                         *self.volatility*np.sqrt(dt)*ran)
            paths[t] = np.maximum(0, paths_[t])
        self.instrument_values = paths
        
        
        
        
        
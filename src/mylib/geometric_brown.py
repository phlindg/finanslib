'''
Created on 17 juni 2017

@author: Phili
'''
from mylib.simulation_class import simulation_class

import numpy as np
from mylib.sndRNG import sn_random_numbers


class geometric_brownian_motion(simulation_class):
    """
    Generates simulated paths based on BSM Brownian motion
    """
    def __init__(self,name,mar_env, corr=False):
        super(geometric_brownian_motion, self).__init__(name, mar_env, corr)
    def update(self, initial_value = None, volatility = None, final_date = None):
        if initial_value != None:
           self.initial_value = initial_value
        if volatility != None:
            self.volatility = volatility
        if final_date != None:
            self.final_date = final_date
        self.instrument_values = None #VARFOR
        
    def generate_paths(self, fixed_seed = False, day_count = 365.):
        if self.time_grid is None:
            self.generate_time_grid()
        #number of dates in time grid
        M = len(self.time_grid)
        #number of paths
        I = self.paths
        #array initalization for path simulation
        paths = np.zeros((M,I))
        paths[0] = self.initial_value
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
            paths[t] = paths[t-1]*np.exp((short_rate-0.5*self.volatility**2)*dt+self.volatility*np.sqrt(dt)*ran)
        self.instrument_values = paths
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
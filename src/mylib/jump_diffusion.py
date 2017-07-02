'''
Created on 17 juni 2017

@author: Phili
'''

import numpy as np
from mylib.sndRNG import sn_random_numbers
from mylib.simulation_class import simulation_class

class jump_diffusion(simulation_class):
    def __init__(self, name, mar_env, corr=False):
        super(jump_diffusion, self).__init__(name, mar_env, corr)
        try:
            #we need more attributes
            self.lamb = mar_env.get_constant("lambda")
            self.mu = mar_env.get_constant("mu")
            self.delt = mar_env.get_constant("delta")
        except:
            print("Error getting stuff for env")
    
    def update(self, initial_value = None, volatility = None,lamb = None, mu = None, delt = None, final_date = None):
        if initial_value != None:
           self.initial_value = initial_value
        if volatility != None:
            self.volatility = volatility
        if lamb != None:
            self.lamb = lamb
        if mu != None:
            self.mu = mu
        if delt != None:
            self.delt = delt
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
            sn1 = sn_random_numbers((1, M, I),fixed_seed = fixed_seed)
        else:
            #if not correlated, use random numbers in market env
            sn1 = self.random_numbers
        #for the jump component
        sn2 = sn_random_numbers((1, M, I),fixed_seed = fixed_seed)
        
        rj = self.lamb* (np.exp(self.mu + 0.5*self.delt**2)-1)
        
        #get short rate for drift of processes
        short_rate = self.discount_curve.short_rate
        for t in range(1,len(self.time_grid)):
            #select the right time slice from the relevant time sets
            if not self.correlated:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:,t,:])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t-1]).days / day_count
            #diff between two dates as year fraction
            poi = np.random.poisson(self.lamb*dt, I)
                #poisson distrubited jump numbers
            paths[t] = paths[t-1]*(np.exp((short_rate-rj-0.5*self.volatility**2)*dt+self.volatility*np.sqrt(dt)*ran)
                                   +(np.exp(self.mu + self.delt*sn2[t])-1)*poi)
        self.instrument_values = paths

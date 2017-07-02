'''
Created on 18 juni 2017

@author: Phili
'''

import numpy as np
from mylib.valuation_class import valuation_class

class valuation_am(valuation_class):
    
    
    def generate_payoff(self, fixed_seed = False):
        try:
            strike = self.strike
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed = fixed_seed)
        time_grid = self.underlying.time_grid
        try:
            time_index_start = int(np.where(time_grid == self.pricing_date)[0])
            time_index_end = int(np.where(time_grid == self.maturity)[0])
        except:
            print ("Maturity not in time grid")
        instrument_values = paths[time_index_start:time_index_end+1]
        try:
            payoff = eval(self.payoff_func)
            return instrument_values, payoff, time_index_start, time_index_end
        except:
            print("error evaling payoff func")
    
    
    def present_value(self,accuracy = 6, fixed_seed = False, bf = 5, full=False):
        instrument_values, inner_values, time_index_start, time_index_end = self.generate_payoff(fixed_seed=fixed_seed)
        #INNER VALUES E ALLTSA PAYOFF
        
        time_list = self.underlying.time_grid[time_index_start:time_index_end+1]
        discount_factors = self.discount_curve.get_discount_factors(time_list, dtobjects = True)
        V = inner_values[-1]
        for t in range(len(time_list)-2,0,-1):
            #derive relevant discount factor for interval
            df = discount_factors[t,1]/discount_factors[t+1,1]
            rg = np.polyfit(instrument_values[t], V*df, bf)
            C = np.polyval(rg, instrument_values[t])
            V = np.where(inner_values[t] > C, inner_values[t], V*df)
        df = discount_factors[0,1]/discount_factors[1,1]
        result = df*np.sum(V)/len(V)
        if full:
            return round(result, accuracy), df*V
        else:
            return round(result,accuracy)
        
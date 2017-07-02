'''
Created on 18 juni 2017

@author: Phili
'''

from mylib.valuation_class import valuation_class
import numpy as np


class valuation_eu(valuation_class):
    
    
    def generate_payoff(self,fixed_seed = False):
        try:
            strike = self.strike
            strike1 = self.strike1
            strike2 = self.strike2
            strike3 = self.strike3
        except:
            pass
        paths = self.underlying.get_instrument_values(fixed_seed = fixed_seed)
        time_grid = self.underlying.time_grid
        try:
            time_index = np.where(time_grid == self.maturity)[0] #??
            time_index = int(time_index)
        except:
            print("Maturity not in time grid of underlying")
        maturity_value = paths[time_index]
        mean_value = np.mean(paths[:time_index],axis=1)
        max_value = np.amax(paths[:time_index], axis=1)[-1]
        min_value = np.amin(paths[:time_index], axis=1)[-1]
        try:
            payoff = eval(self.payoff_func) #bra sak, las pa mer om den!
            return payoff
        except:
            print("Error evaluating payoff")
    
    def present_value(self, accuracy = 6, fixed_seed = False, full=False):
        
        cash_flow = self.generate_payoff(fixed_seed=fixed_seed)
        
        discount_factor = self.discount_curve.get_discount_factors(
            (self.pricing_date, self.maturity))[0,1]
        result = discount_factor *np.sum(cash_flow)/len(cash_flow)
        if full:
            return round(result, accuracy), discount_factor * cash_flow
        else:
            return round(result,accuracy)
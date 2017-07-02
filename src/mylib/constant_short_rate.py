'''
Created on 15 juni 2017

@author: Phili
'''
import numpy as np
def get_year_deltas(date_list, day_count=365.):
    """
    Returns a vector with day deltas in years. Floats.
    Parameters:
        collection of datetime objects
    """
    start = date_list[0]
    delta_list = [(date-start).days/day_count for date in date_list]
    return np.array(delta_list)

class constant_short_rate:
    """
    name:
        name of the object
    short_rate: float > 0
        constant rate for discouning
    """
    def __init__(self,name,short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError("Short rate negative")
    def get_discount_factors(self, date_list, dtobjects = True):
        if dtobjects == True:
            dlist = get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * np.sort(-dlist))
        return np.array((date_list,dflist)).T
    
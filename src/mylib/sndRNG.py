'''
Created on 17 juni 2017

@author: Phili
'''

import numpy as np

def sn_random_numbers(shape, antithetic = False, moment_matching = True, fixed_seed = False):
    """
    Returns a array with standard normally distrubited numbers
    
    Parameters
    ============================
    shape: tuple (o,n,m)
        Array with shape (o,n,m)
    Antithetic: Boolean
        Generation of antithetic variates.
    moment_matching: Boolean
        matching of first and second moments
    fixed_seed: Boolean
        Flag to fix the seed
    """
    if fixed_seed:
        np.random.seed(1000)
    if antithetic:
        ran = np.random.standard_normal((shape[0],shape[1],shape[2]/2))
        ran = np.concatenate((-ran, ran),axis=2)
    else:
        ran = np.random.standard_normal(shape)
    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran/np.std(ran)
    if shape[0] == 1:
        return ran[0]
    else:
        return ran
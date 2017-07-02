'''
Created on 15 juni 2017

@author: Phili
'''

"""
Approximation:
    Regression and interpolation are used very much in finance
    If you have sorted and not so noicy data you should use interpolation (like cubic splines)
    If you have that, which is usually the case, you should use regression.
Convex Optimization:
    
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import sympy as sy

def approximaton():
    def f(x):
        return np.sin(x) + 0.5*x
    x = np.linspace(-2*np.pi, 2*np.pi, 50)
    reg = np.polyfit(x, f(x), deg=1)
    ry = np.polyval(reg,x)
    
    plt.plot(x,f(x),'b')
    plt.plot(x,ry, 'r.')
    plt.grid(True)
    plt.show()
def convex_opti():
    output = True
    def fo(x,y):
        z = np.sin(x)+0.05*x**2 + np.sin(y) + 0.05*y**2
        if output == True:
            print("%8.4f    %8.4f    %8.4f"    %    (x,    y,    z))
        return z
def symbolic_comp():
    x = sy.Symbol('x')
    y = sy.Symbol('y')
    f = 3+sy.sqrt(x) - 4**2
    g = sy.solve(f)
    i = sy.integrate(f)
    Fb = i.subs(x,9.5).evalf()
    Fa = i.subs(x,0.5).evalf()
    exact_integral_value = Fb - Fa
    derivera = i.diff()
symbolic_comp()
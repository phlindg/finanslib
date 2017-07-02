'''
Created on 19 juni 2017

@author: Phili
'''

class derivates_pos:
    def __init__(self, name, quantity, underlying, mar_env, otype, payoff_func):
        self.name = name
        self.quantity = quantity
        self.underlying = underlying
        self.mar_env = mar_env
        self.otype = otype
        self.payoff_func = payoff_func
        
    def get_info(self):
        print("NAME")
        print (self.name)
        print("QUANTITY")
        print(self.quantity)
        print("UNDERLYING")
        print(self.underlying)
        print("**MARKET ENV **")
        print("CONSTANTS")
        for key, value in self.mar_env.constants.items():
            print (key,value)
        print("LISTS")
        for key, value in self.mar_env.lists.items():
            print (key,value)
        print("CURVES")
        for key, value in self.mar_env.curves.items():
            print (key,value)
        print("OPTION TYPE")
        print(self.otype)
        print("PAYOFF FUNCTION")
        print(self.payoff_func)
        
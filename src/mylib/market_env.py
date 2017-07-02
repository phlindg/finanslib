'''
Created on 17 juni 2017

@author: Phili
'''


class market_env:
    def __init__(self,name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}
    def add_constant(self, key,constant):
        self.constants[key] = constant
    def get_constant(self, key):
        return self.constants[key]
    def add_list(self, key,list_obj):
        self.lists[key] = list_obj
    def get_list(self,key):
        return self.lists[key]
    def add_curve(self,key,curve):
        self.curves[key] = curve
    def get_curve(self,key):
        return self.curves[key]
    def add_env(self,env):
        #Overrides everything
        for key in env.constants:
            self.constants[key] = env.constants[key]
        for key in env.lists:
            self.lists[key] = env.lists[key]
        for key in env.curves:
            self.curves[key] = env.curves[key]

"""
Whats really good with this one is that its easy to save certain data in an instance of the class.
"""
'''
Created on 18 juni 2017

@author: Phili
'''

class valuation_class:
    def __init__(self, name, underlying, mar_env, payoff_func = ""):
        try:
            self.name = name
            self.pricing_date = mar_env.pricing_date
            try:
                #pga strike optional
                self.strike = mar_env.get_constant("strike")
                self.strike1 = mar_env.get_constant("strike1")
                self.strike2 = mar_env.get_constant("strike2")
                self.strike3 = mar_env.get_constant("strike3")
            except:
                pass
            self.maturity = mar_env.get_constant("maturity")
            self.currency = mar_env.get_constant("currency")
            
            #parameters from simulation object
            self.frequency = underlying.frequency
            self.paths = underlying.paths
            self.discount_curve = underlying.discount_curve
            
            self.payoff_func = payoff_func
            self.underlying = underlying
            
            #satt pricing_date och maturity till underlying
            self.underlying.special_dates.extend([self.pricing_date, self.maturity])
        except:
            print("Error getting stuff for env (valuation)")
    
    def update(self, initial_value = None, volatility = None,strike=None, maturity = None):
        if initial_value is not None:
           self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity
            #lagg till ny om den ej e i time_grid
            if not maturity in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None
    
    def delta(self,interval = None, accuracy = 4):
        if interval is None:
            interval = self.underlying.initial_value / 50.
        
        value_left = self.present_value(fixed_seed = True)
        initial_del = self.underlying.initial_value + interval
        self.underlying.update(initial_value = initial_del)
        
        value_right = self.present_value(fixed_seed = True)
        self.underlying.update(initial_value = initial_del - interval)
        delta = (value_right-value_left) / interval
        
        #correction
        if delta > 1:
            return 1
        elif delta < -1:
            return -1
        else:
            return round(delta,accuracy)
    
    def vega(self, interval = 0.01, accuracy = 4):
        if interval < self.underlying.volatility / 50.:
            interval = self.underlying.volatility/50.
        value_left = self.present_value(fixed_seed = True)
        vola_del = self.underlying.volatility + interval
        self.underlying.update(volatility = vola_del)
        value_right = self.present_value(fixed_seed = True)
        self.underlying.update(volatility = vola_del - interval)
        vega = (value_right - value_left) / interval
        return round(vega, accuracy)
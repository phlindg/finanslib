'''
Created on 15 juni 2017

@author: Phili
'''

from mylib import *
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
def brown():
    me_gbm = market_env("me_gbm", dt.datetime(2016,1,1))
    me_gbm.add_constant("initial_value", 36.)
    me_gbm.add_constant("volatility", 0.2)
    me_gbm.add_constant("final_date", dt.datetime(2016,12,31))
    me_gbm.add_constant("currency", "EUR")
    me_gbm.add_constant("frequency", "M")
    me_gbm.add_constant("paths", 1000)
    csr = constant_short_rate("csr", 0.05)
    me_gbm.add_curve("discount_curve", csr)
    
    gbm = geometric_brownian_motion("gbm", me_gbm)
    gbm.generate_time_grid()
    paths = gbm.get_instrument_values()
    plt.plot(gbm.time_grid, paths[:,:10])
    plt.show()
    
def jump():
    me_jd = market_env("me_jd", dt.datetime(2016,1,1))
    me_jd.add_constant("initial_value", 36.)
    me_jd.add_constant("volatility", 0.2)
    me_jd.add_constant("final_date", dt.datetime(2016,12,31))
    me_jd.add_constant("currency", "EUR")
    me_jd.add_constant("frequency", "M")
    me_jd.add_constant("paths", 10000)
    csr = constant_short_rate("csr", 0.05)
    me_jd.add_curve("discount_curve", csr)
    me_jd.add_constant("lambda", 0.3)
    me_jd.add_constant("mu", -0.75)
    me_jd.add_constant("delta", 0.1)
    
    jd = jump_diffusion("jd", me_jd)
    jd.generate_time_grid()
    paths = jd.get_instrument_values()
    print(paths)
    plt.plot(jd.time_grid, paths[:,:10])
    plt.show()
    
def sqrt_diff():
    me_srd = market_env("me_srd", dt.datetime(2015,1,1))
    me_srd.add_constant("initial_value", 0.25)
    me_srd.add_constant("volatility", 0.05)
    me_srd.add_constant("final_date", dt.datetime(2016,12,31))
    me_srd.add_constant("currency", "EUR")
    me_srd.add_constant("frequency", "W")
    me_srd.add_constant("paths", 10000)
    me_srd.add_constant("kappa", 4.0)
    me_srd.add_constant("theta", 0.2)
    me_srd.add_curve("discount_curve", constant_short_rate('r', 0.0))
    
    srd = sqrt_diffusion('srd', me_srd)
    paths = srd.get_instrument_values()
    srd.generate_time_grid()
    plt.plot(srd.time_grid, paths[:,:10])
    plt.axhline(me_srd.get_constant("theta"), color="r")
    plt.show()

def eval_opt():
    me_gbm = market_env("me_gbm", dt.datetime(2015,1,1))
    me_gbm.add_constant("initial_value", 36.)
    me_gbm.add_constant("volatility", 0.2)
    me_gbm.add_constant("final_date", dt.datetime(2016,12,31))
    me_gbm.add_constant("currency", "EUR")
    me_gbm.add_constant("frequency", "W")
    me_gbm.add_constant("paths", 50000)
    csr = constant_short_rate("csr", 0.06)
    me_gbm.add_curve("discount_curve", csr)
    gbm = geometric_brownian_motion("gbm", me_gbm)
    
    me_call = market_env("me_call", me_gbm.pricing_date)
    me_call.add_constant("strike", 40)
    me_call.add_constant("maturity", dt.datetime(2015,12,31))
    me_call.add_constant("currency", "EUR")
    payoff_func = "np.maximum(maturity_value - strike, 0)"
    payoff_func_am = "np.maximum(strike - instrument_values, 0)"
    pay = "maturity_value**2"
    eur_call = valuation_eu("eur_call", underlying = gbm, mar_env = me_call, payoff_func = pay)
    eur_call_pos = derivates_pos(
        name = "eur_call_pos",
        quantity = 3,
        underlying = "gbm",
        mar_env = me_call,
        otype = "Eu Call",
        payoff_func = payoff_func)
    eur_call_pos.get_info()
    

def call_opt(strike):
    return "np.maximum(maturity_value - "+strike+", 0)"

def portfolio_c():
    
    
    
    me_gbm = market_env("me_gbm", dt.datetime(2015,1,1))
    me_gbm.add_constant("initial_value", 36.)
    me_gbm.add_constant("volatility", 0.2)
    me_gbm.add_constant("final_date", dt.datetime(2016,12,31))
    me_gbm.add_constant("currency", "EUR")
    me_gbm.add_constant("frequency", "W")
    me_gbm.add_constant("paths", 50000)
    csr = constant_short_rate("csr", 0.06)
    me_gbm.add_curve("discount_curve", csr)
    me_gbm.add_constant("model", "gbm")
    
    me_jd = market_env("me_jd", dt.datetime(2015,1,1))
    me_jd.add_constant("lambda", 0.3)
    me_jd.add_constant("mu", -0.75)
    me_jd.add_constant("delta", 0.1)
    me_jd.add_env(me_gbm)
    #needed for portfolio valuation
    me_jd.add_constant("model", "jd")
    
    gbm = geometric_brownian_motion("gbm", me_gbm)
    jd = jump_diffusion("jd", me_jd)
    
    me_eur_call = market_env("me_eur_call", me_jd.pricing_date)
    me_eur_call.add_constant("maturity", dt.datetime(2015,6,30))
    me_eur_call.add_constant("strike", 38.)
    me_eur_call.add_constant("currency", "EUR")
    
    me_am_put = market_env("me_am_put", me_gbm.pricing_date)
    me_am_put.add_constant("maturity", dt.datetime(2015,6,30))
    me_am_put.add_constant("strike", 40.)
    me_am_put.add_constant("currency", "EUR")
    
    butterfly = market_env("butterfly", me_gbm.pricing_date)
    butterfly.add_constant("maturity", dt.datetime(2015,6,30))
    butterfly.add_constant("currency", "EUR")
    
    straddle = market_env("straddle", me_gbm.pricing_date)
    straddle.add_constant("maturity", dt.datetime(2015,6,30))
    straddle.add_constant("strike", 40.)
    straddle.add_constant("currency", "EUR")
    
    asset = market_env("asset_or_nothing", me_gbm.pricing_date)
    asset.add_constant("maturity", dt.datetime(2016,6,30))
    asset.add_constant("strike", 40.)
    asset.add_constant("currency", "EUR")
    
    call = "np.maximum(maturity_value - strike, 0)"
    put = "np.maximum(strike - maturity_value, 0)"
    payoff_func_eu = call_opt("strike")
    payoff_func_am = "np.maximum(strike - instrument_values,0)"
    payoff_func_butterfly = "np.maximum(maturity_value - 0.95*36., 0) - np.maximum(maturity_value - 36., 0) - np.maximum(maturity_value - 36., 0) + np.maximum(maturity_value - 1.05*36., 0)"
    payoff_func_straddle = "np.maximum(maturity_value - strike, 0) + np.maximum(strike-maturity_value,0)"
    payoff_func_asset_or_nothing = "maturity_value * np.maximum(maturity_value - strike, 0)/np.maximum(maturity_value - strike, 0.00000000000001)"
    eur_call_pos = derivates_pos(
        name = "eur_call_pos",
        quantity = 1,
        underlying = "gbm",
        mar_env = me_eur_call,
        otype ="European",
        payoff_func = payoff_func_eu)
    am_put_pos = derivates_pos(
        name = "am_put_pos",
        quantity = 1,
        underlying = "gbm",
        mar_env = me_am_put,
        otype = "American",
        payoff_func = payoff_func_am
        )
    butterfly_pos = derivates_pos(
        name="butterfly",
        quantity=1,
        underlying = "gbm",
        mar_env = butterfly,
        otype="European",
        payoff_func = payoff_func_butterfly)
    straddle_pos = derivates_pos(
        name = "straddle",
        quantity = 1,
        underlying = "gbm",
        mar_env = straddle,
        otype="European",
        payoff_func = payoff_func_straddle)
    asset_or_nothing_pos = derivates_pos(
        name="asset_or_nothing",
        quantity=1,
        underlying ="jd",
        mar_env = asset,
        otype="European",
        payoff_func = payoff_func_asset_or_nothing)
    underlyings = {"gbm":me_gbm,
                   "jd": me_jd}
    positions = {"am_put_pos": am_put_pos,
                 "eu_call_pos": eur_call_pos,
                 "butterfly": butterfly_pos,
                 "straddle" : straddle_pos,
                 "asset_or_nothing": asset_or_nothing_pos}
    val_env = market_env("general", me_gbm.pricing_date)
    val_env.add_constant("frequency", "W")
    val_env.add_constant("paths", 250000)
    val_env.add_constant("starting_date", val_env.pricing_date)
    val_env.add_constant("final_date", val_env.pricing_date)
    val_env.add_curve("csr", csr)
    
    corr = [['gbm', 'jd', 0]]
    
    portfolio = derivatives_portfolio(
        name="portfolio",
        positions=positions,
        val_env = val_env,
        assets = underlyings,
        correlations = corr,
        fixed_seed = True
        )
    return portfolio
def plotta():
    pf = portfolio_c()

    path_no = 777
    path_gbm = pf.underlying_objects['gbm'].get_instrument_values()[:,path_no]
    path_jd = pf.underlying_objects['jd'].get_instrument_values()[:,path_no]
    
    plt.plot(pf.time_grid, path_gbm, 'r', label = 'gbm')
    plt.plot(pf.time_grid, path_jd, 'g', label = 'jd')
    plt.xticks(rotation = 30)
    plt.legend(loc=0); plt.grid()
    plt.show()
    
    

plotta()
    
    
    
    
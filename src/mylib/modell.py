'''
Created on 28 juni 2017

@author: Phili
'''


#https://github.com/yhilpisch/eurexas/tree/master/vstoxx/data
#https://www.quandl.com/databases/ZFA/documentation/database-overview

from mylib import *
import pandas as pd
import pandas_datareader as pdr
import fix_yahoo_finance
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as spo
import calendar
import h5py
import quandl
quandl.ApiConfig.api_key = "ZazfayyxoPXJCJjhBi2_"

#url = 'https://www.stoxx.com/document/Indices/Current/HistoricalData/h_vstoxx.txt'
#vstoxx_index = pd.read_csv(url, index_col = 0, header = 2, parse_dates = True, dayfirst = True)

#data = vstoxx_index[('2013/12/31' < vstoxx_index.index) & (vstoxx_index.index < '2014/4/1')]


def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
    datas = map (data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
def get1(stock,startdate,enddate):
    return pdr.get_data_yahoo(stock, start=startdate, end=enddate)


def third_friday(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1)+2)%7
    return dt.datetime(date.year, date.month, day)


    

    
i = 0

def test():
    req = pd.read_csv("data\msft.csv")
    stock = req[(req["date"] > "2015-11-15")
                & (req["date"] < "2016-06-30")]
    
    
    
    
    
    
    opts = pd.read_csv("data\opts.csv")
    opts_msft = opts[opts.ticker == "MSFT"]
    data = opts_msft[['stkPx', 'expirDate', 'strike', 'cValue','trade_date']]
    columns = ["STOCK PRICE", "MATURITY",  "STRIKE", "CALL VALUE", "DATE"]
    data.columns = columns
    test = []
    for opt in data["MATURITY"]:
        test.append(int(opt[5:7]))
    data.insert(5, "EXP_MONTH", test)
    data = data[(data["MATURITY"] < "2016-06-25")]
    third_fridays = {}
    for month in set(data["EXP_MONTH"]):
        third_fridays[month] = third_friday(dt.datetime(2016, month, 1))
   
    tf = lambda x: third_friday(x)
    
     
    
    pricing_date = dt.datetime(2015,11,17)
    pricing_date_str = "2015-11-17"
    initial_value = stock[stock["date"] == "2015-11-17"].iloc[0]["close"]
   
    maturity = third_fridays[6]
    maturity_str = str(maturity)
    tol = 0.1
    
    option_selection = data[(data["DATE"].values == pricing_date_str)
                            & (data["MATURITY"].values == maturity_str[0:10])
                            & (data["STRIKE"].values > (1-tol)*initial_value)
                            & (data["STRIKE"].values < (1+tol)*initial_value)]
    
    
    
    me_stocks = market_env("me_stocks", pricing_date)
    me_stocks.add_constant("initial_value", initial_value)
    me_stocks.add_constant("final_date", maturity)
    me_stocks.add_constant("currency", "USD")
    me_stocks.add_constant("frequency", "B")
    me_stocks.add_constant("paths", 10000)
    csr = constant_short_rate("csr", 0.01)
    me_stocks.add_curve("discount_curve", csr)
    #kaliberas seanre
    me_stocks.add_constant("lambda", 0.3)
    me_stocks.add_constant("mu", -0.75)
    me_stocks.add_constant("delta", 0.1)
    vol_est = stock.std() * np.sqrt(len(stock)/252.)
    vol_est = vol_est["close"]
    me_stocks.add_constant("volatility", vol_est)
    me_stocks.add_constant("strike", initial_value) #<----- TO BE CHANGED.
    me_stocks.add_constant("maturity", maturity)
    
    
    stocks_model = jump_diffusion("stocks_model", me_stocks)
    
    payoff_func = "np.maximum(maturity_value - strike, 0)"
    stocks_eu_call = valuation_eu("stocks_eu_call", stocks_model, me_stocks, payoff_func)
    pv = stocks_eu_call.present_value()
    print(pv)
    option_models = {}
    for option in option_selection.index:
        strike = option_selection["STRIKE"].ix[option]
        
        me_stocks.add_constant("strike", strike)
        option_models[option] = valuation_eu(
            "eur_call_%d" % strike,
            stocks_model,
            me_stocks,
            payoff_func)
    for opt in option_models.values():
        print(opt.name, opt.present_value())
    def calc_model_values(p0):
        lamb, mu, delta, vol = p0
        stocks_model.update(lamb = lamb, mu = mu, delt = delta, volatility = vol)
        model_values = {}
        for option in option_models:
            model_values[option] = option_models[option].present_value(fixed_seed = True)
        return model_values
    
    def mean_squared_error(p0):
        global i
        model_values = np.array(list(calc_model_values(p0).values()))
        
        market_values = option_selection["CALL VALUE"].values
        option_diffs = model_values - market_values
        MSE = np.sum(option_diffs ** 2)/len(option_diffs)
        if i % 20 == 0:
            if i == 0:
                print("%4s %6s %6s %6s %6s --> %6s" % ("i","lambda","mu", "delta", "vola", "MSE"))
            print("%4s %6s %6s %6s %6s --> %6s" % (i,p0[0],p0[1],p0[2],p0[3], MSE))
        i+=1
        return MSE
    
    #print(calc_model_values((0.3, -0.75, 0.1, vol_est)))
    i = 0
    opt_global = spo.brute(mean_squared_error,
                           ((1.0, 10.0, 1),
                          (-2.0, 2.0, 0.4),
                           (0.0, 0.5, 0.05),
                           (0.5, 5.51, 0.5)),
                         finish = None)
    hmm = (2.55, 1.5, 0.45, 5.5)
    i = 0
    #opt_local = spo.fmin(mean_squared_error, 
     #                    hmm, xtol = 0.0001, ftol = 0.00001, 
      #                   maxiter = 10000, maxfun = 1000)
    hmm = (2.43911542147, 1.51858536661, 0.454485565729, 5.79759553969)
    option_selection["MODEL"] = np.array(list(calc_model_values(hmm).values()))
    option_selection["ERRORS"] = option_selection["MODEL"] - option_selection["CALL VALUE"]
    print(option_selection[["MODEL", "CALL VALUE", "ERRORS"]])
    
   
test()





    
    
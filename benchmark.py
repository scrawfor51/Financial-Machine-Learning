#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:50:19 2022

@author: Stephen
"""

import pandas as pd
import datetime
import math as m
import numpy as np
import os


class Benchmarker: 
    
    def __init__(self, start_date ='2018-01-01', end_date = '2019-12-31', symbols =['DIS'], starting_value = 100000, include_spy=False):
    
        self.price_data = self.get_data(start_date, end_date, symbols, include_spy)
        self.start = start_date
        self.end = end_date
        self.starting_value = starting_value
        self.baseline = self.price_data.copy()
        self.baseline.loc[:,:] = np.NaN
        self.baseline.iloc[0,0] = 1000
        self.baseline_portfolio = self.assess_strategy_dataframe(self.baseline, self.start, self.end, starting_value = self.starting_value, fixed_cost = 0, floating_cost = 0)
        self.calc_portfolio(self.baseline_portfolio)
        
    """
    Helper function to read data files of requested stocks and extract date range 
    
    @param start_date: The start of the date range you want to extract
    @parem end_date: The end of the date range you want to extract
    @param symbols: A list of stock ticker symbols
    @param column_name: A single data column to retain for each symbol 
    @param include_spy: Controls whether prices for SPY will be retained in the output 
    @return A dataframe of the desired stocks and their data
    """
    def get_data(self, start_date, end_date, symbols, column_name = 'Adj Close', include_spy=True):
        
    
        standardized_symbols = []
        #Catch lowercase symbols
        
        if include_spy:
            standardized_symbols.append('SPY')
        for symbol in symbols:
            standardized_symbols.append(symbol.upper())
            
            
        queried_data = pd.DataFrame(columns=standardized_symbols) #make an empty dataframe as a series of ticker name
        data_path = './data'
        
        for file in os.listdir(data_path):
             if file[:file.find('.')] in standardized_symbols: 
                 df = pd.read_csv(os.path.join(data_path, file), index_col='Date', parse_dates=True, float_precision=(None))
                 df = df.loc[start_date : end_date]
                 queried_data[file[:file.find('.')]] = df[column_name]
                 
        return(queried_data)
    
    
    def assess_strategy_dataframe(self, trades, start_date, end_date, starting_value = 1000000, fixed_cost = 9.95, floating_cost = 0.005):
    
        #Get traded symbol
        symbol = trades.columns
    
        #get stock data
        stocks_vals = self.get_data(start_date, end_date, symbol, include_spy=False)
       
        #shares = stocks_vals.index
        shares = pd.DataFrame(index=stocks_vals.index)
        
        cash_val = pd.DataFrame(index=stocks_vals.index)
        cash_val.loc[start_date:, "Cash"] = starting_value
      
        shares.loc[start_date:, symbol] = 0
    
    
        
        for day in trades.index:
            
            trade_shares = trades.loc[day,symbol].values[0]
            total_transaction_value = trade_shares * stocks_vals.loc[day, symbol].values[0]
            
    
            if trade_shares > 0:
                shares.loc[day : , symbol] += trade_shares
                cash_val.loc[day :, "Cash"] -=  total_transaction_value*(1 + floating_cost) + fixed_cost
                
            elif trade_shares < 0:
                shares.loc[day :, symbol] += trade_shares
                cash_val.loc[day : , "Cash"] -=  total_transaction_value*(1 - floating_cost) - fixed_cost
       
        portfolio_val = pd.DataFrame(index=stocks_vals.index)
        
        portfolio_val['Portfolio'] = (shares.values * stocks_vals.values)+ cash_val
        
        return portfolio_val
    
    def calc_portfolio(self, portfolio_val, starting_value = 200000, risk_free_rate = 0.0, sample_freq = 252):
        
        portfolio = portfolio_val.copy()
        end_value = portfolio_val["Portfolio"].iloc[-1]
        average_daily_return = ((portfolio["Portfolio"]/ portfolio["Portfolio"].shift()) - 1).mean()
        stdev_daily_return = ((portfolio_val["Portfolio"]/ portfolio_val["Portfolio"].shift()) - 1).std()
        cumulative_return = end_value/(starting_value) 
        sharpe_ratio = m.sqrt(sample_freq) * ((average_daily_return - risk_free_rate)/ stdev_daily_return)
    
        print("Sharpe Ratio: ", sharpe_ratio)    
        print("Volatility (stdev of daily returns): ", stdev_daily_return)
        print("Average Daily Return: ", average_daily_return)
        print("Cumulative Return: ", cumulative_return)
        print("End Value: ", end_value)
    
    
     

    

 
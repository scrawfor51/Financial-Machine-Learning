#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:09:28 2022

Backtester -- Given a CSV file of trades simulates the resulting portfolio changes over time and reports the overall performance 
@author: Stephen Crawford
"""
import os
import pandas as pd 
from matplotlib import pyplot as plt 
import math


"""
Helper function to read data files of requested stocks and extract date range 

@param start_date: The start of the date range you want to extract
@parem end_date: The end of the date range you want to extract
@param symbols: A list of stock ticker symbols
@param column_name: A single data column to retain for each symbol 
@param include_spy: Controls whether prices for SPY will be retained in the output 
@return A dataframe of the desired stocks and their data
"""
def get_data(start_date, end_date, symbols, column_name = 'Adj Close', include_spy=True):
    
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



"""
Given a csv file path, a starting value of money, a fixed trade cost, and a floating trade cost, determine the efficacy of a trading strategy. 


@pararm trade_file: A relative file path to a csv of trades of the form: Date, Symbol, Order Direction, Number of Shares
@param starting_value: The amount of cash you start with before making any trades
@param fixed_cost: The fixed fee in dollars that you are charged for every transaction
@param floating_cost: A percentage cost you are charged on the total value of any transaction (not including the fixed_cost fee)
@return: A 1-column DataFrame of the portfolio dollar value for every day in the range of dates for which the strategy made trades. 
"""
def assess_strategy(trade_file = "./trades/trades.csv", starting_value = 1000000, fixed_cost = 9.95, floating_cost = 0.005, leverage=None):
    
    
    trade_history = pd.read_csv(trade_file, parse_dates=True)
    start_date = trade_history.iloc[0, 0] #First trade date
    end_date = trade_history.iloc[-1, 0] #Last trade date
    tickers = trade_history['Symbol'].unique() #All the stocks we care about 
  
    daily_stock_prices = get_data(start_date, end_date, tickers, include_spy=(False))
    daily_stock_prices['CASH'] = 1 # Set the value of CASH to $1/share
    
    daily_portfolio_values = pd.DataFrame(columns=['CASH'])
    daily_portfolio_values = pd.concat((daily_stock_prices, daily_portfolio_values), axis=0)
    daily_portfolio_values.loc[:, tickers] = 0
    daily_portfolio_values['CASH'] = starting_value
    

    for index, row in trade_history.iterrows(): 

        date = row['Date']
        count = row['Shares']
        sym = row['Symbol']
       
        
        if leverage: 
             
            pre_trade_at_risk = sum(abs(daily_portfolio_values.loc[date, daily_portfolio_values.columns != 'CASH']) * daily_stock_prices.loc[date, daily_portfolio_values.columns != 'CASH'])
            pre_trade_liquid = sum(daily_portfolio_values.loc[date, :] * daily_stock_prices.loc[date, :])
            pre_trade_leverage = (pre_trade_at_risk/pre_trade_liquid)
         
            
            if row['Direction'] == 'BUY':
                post_trade_at_risk = pre_trade_at_risk + (daily_stock_prices.loc[date, sym] * count) 
                #print("Proposed trade to buy ", sym, " at ", daily_stock_prices.loc[date, sym], " curent holdings of ", sym, " ", daily_portfolio_values.loc[date, sym], " at risk capital before ", pre_trade_at_risk, " is now ", post_trade_at_risk)
                post_trade_liquid = pre_trade_liquid - fixed_cost - floating_cost*(daily_stock_prices.loc[date, sym] * count)
                
            if row['Direction'] == 'SELL':
                post_trade_at_risk = pre_trade_at_risk - (daily_stock_prices.loc[date, sym] * (daily_portfolio_values.loc[date, sym] - count))
                #print("Proposed trade to sell ", sym, " at ", daily_stock_prices.loc[date, sym], " curent holdings of ", sym, " ", daily_portfolio_values.loc[date, sym], " at risk capital before ", pre_trade_at_risk, " is now ", post_trade_at_risk)
                post_trade_liquid = pre_trade_liquid - fixed_cost - floating_cost*(daily_stock_prices.loc[date, sym] * count)
                
            if row['Direction'] == 'DEPOSIT': # Simulate adding cash to your portfolio 
                post_trade_at_risk = pre_trade_at_risk
                post_trade_liquid = pre_trade_liquid + (daily_stock_prices.loc[date, 'CASH'] * count)
                     
            post_trade_leverage = (post_trade_at_risk/post_trade_liquid)
     
            
            if (abs(post_trade_leverage) <= leverage and post_trade_leverage >= 0) or (abs(post_trade_leverage) <= abs(pre_trade_leverage) and post_trade_leverage >= 0) :
                
                if row['Direction'] == 'BUY':
                    daily_portfolio_values.loc[date:,'CASH'] =  (daily_portfolio_values.loc[date,'CASH']) - (daily_stock_prices.loc[date, sym] * count) - fixed_cost - floating_cost*(daily_stock_prices.loc[date, sym] * count)
                    daily_portfolio_values.loc[date:, sym] = daily_portfolio_values.loc[date, sym] + count
                    #print("Bought ", count, " shares of ", sym, " at ", (daily_stock_prices.loc[date, sym]), " spent ", (daily_stock_prices.loc[date, sym] * count), " charged ", fixed_cost + floating_cost*(daily_stock_prices.loc[date, sym] * count), " account balance now: ", (daily_portfolio_values.loc[date,'CASH']))
                if row['Direction'] == 'SELL':
                    daily_portfolio_values.loc[date:,'CASH'] =  (daily_portfolio_values.loc[date,'CASH']) + (daily_stock_prices.loc[date, sym] * count) - fixed_cost - floating_cost*(daily_stock_prices.loc[date, sym] * count)
                    daily_portfolio_values.loc[date:, sym] =  daily_portfolio_values.loc[date, sym] - count
                    #print("Sold ", count, " shares of ", sym, " at ", (daily_stock_prices.loc[date, sym]), " earned ", (daily_stock_prices.loc[date, sym] * count), " charged ", fixed_cost + floating_cost*(daily_stock_prices.loc[date, sym] * count), " account balance now: ", (daily_portfolio_values.loc[date,'CASH']))
                if row['Direction'] == 'DEPOSIT':# Simulate adding cash to your portfolio 
                    daily_portfolio_values.loc[date:,'CASH'] =  (daily_portfolio_values.loc[date,'CASH']) + (daily_stock_prices.loc[date, 'CASH'] * count)
                         
            else: 
                #print("Trade prevented by broker.")
                continue
    
        else: 
            if row['Direction'] == 'BUY':
                daily_portfolio_values.loc[date:,'CASH'] =  (daily_portfolio_values.loc[date,'CASH']) - (daily_stock_prices.loc[date, sym] * count) - fixed_cost - floating_cost*(daily_stock_prices.loc[date, sym] * count)
                daily_portfolio_values.loc[date:, sym] = daily_portfolio_values.loc[date, sym] + count
                #print("Bought ", count, " shares of ", sym, " at ", (daily_stock_prices.loc[date, sym]), " spent ", (daily_stock_prices.loc[date, sym] * count), " charged ", fixed_cost + floating_cost*(daily_stock_prices.loc[date, sym] * count), " account balance now: ", (daily_portfolio_values.loc[date,'CASH']))
            if row['Direction'] == 'SELL':
                daily_portfolio_values.loc[date:,'CASH'] =  (daily_portfolio_values.loc[date,'CASH']) + (daily_stock_prices.loc[date, sym] * count) - fixed_cost - floating_cost*(daily_stock_prices.loc[date, sym] * count)
                daily_portfolio_values.loc[date:, sym] =  daily_portfolio_values.loc[date, sym] - count
                #print("Sold ", count, " shares of ", sym, " at ", (daily_stock_prices.loc[date, sym]), " earned ", (daily_stock_prices.loc[date, sym] * count))
            if row['Direction'] == 'DEPOSIT':# Simulate adding cash to your portfolio 
                daily_portfolio_values.loc[date:,'CASH'] =  (daily_portfolio_values.loc[date,'CASH']) + (daily_stock_prices.loc[date, 'CASH'] * count)
                    
                
    scaled_portfolio = daily_portfolio_values.mul(daily_stock_prices)
   
    daily_portfolio_values['PORT'] = scaled_portfolio.sum(axis=1)
    return daily_portfolio_values['PORT']



"""
Main function to analyze a portfolio's performance.
Takes in a set of tickers, an list of corresponding allocations, and a date range and calculates common metrics.

@param daily_portfolio_values: A dataframe of the results of the portfolio's tradiing choices
@param risk_free_rate: The expected level of returns for a 'risk-free' investment over a period of time
@parma benchmark: The benchmark index to be used for comparison. 
@param sample_freq: The sample frequency per year we are interested in standardizing to (accounts for inherent volatility in morre commonly traded stocks)
@param plot_returns: Whether we want to see a plot of SPY vs. the portfolio over the time period
@return cumulative_return: Zero-based value of the portfolio at the end date
@return average_daily_return: Zero-based average of daily portfolio returns
@return stdev_daily_return: Standard deviation of daily portfolio returns
@return sharpe_ratio: Annualized value indicating returns vs. risk
@return end_value: Dollar total of portfolio at end date
"""
def assess_portfolio (daily_portfolio_values, risk_free_rate=0.0, benchmark=['^SPX'],
                      sample_freq=252, plot_returns=False, display_output=False):
    
   data = daily_portfolio_values
   start_date = data.index[0]
   end_date = data.index[-1]
   bench_values = get_data(start_date, end_date, benchmark, include_spy=(False))
   data = pd.concat((data, bench_values), axis=1)

   cumulative_returns = (data/data.iloc[0] - 1)  # Get cumulative returns
   cumulative_return_port = cumulative_returns.iloc[-1]['PORT']
   cumulative_return_bench = cumulative_returns.iloc[-1][benchmark]
   
   average_daily_return_port = ((data/data.shift()).mean(axis=0) - 1)['PORT']# Get average daily returns 
   average_daily_return_bench = ((data/data.shift()).mean(axis=0) - 1)[benchmark]# Get average daily returns 
  
   stdev_daily_return_port = (data/data.shift()).std(axis=0)['PORT']# Get standard devaition of daily returns 
   stdev_daily_return_bench = (data/data.shift()).std(axis=0)[benchmark]
  
   sharpe_ratio_port = ((data/data.shift() - 1 - risk_free_rate)/stdev_daily_return_port).mean() # calculate sharpe ratio
   sharpe_ratio_port = math.sqrt(sample_freq) * sharpe_ratio_port #Annualize sharpe ratio
   sharpe_ratio_port = sharpe_ratio_port['PORT'] # Get sharpe val\ sharpe_ratio_port = ((data/data.shift() - 1 - risk_free_rate)/stdev_daily_return_port).mean() # calculate sharpe ratio
   
   sharpe_ratio_bench = ((data/data.shift() - 1 - risk_free_rate)/stdev_daily_return_bench).mean()
   sharpe_ratio_bench = math.sqrt(sample_freq) * sharpe_ratio_bench #Annualize sharpe ratio
   sharpe_ratio_bench = sharpe_ratio_bench[benchmark] # Get sharpe val
   sharpe_ratio_bench = sharpe_ratio_bench[benchmark]
   
   end_value_port = data.iloc[-1]['PORT'] #Get target val
   end_value_bench = data.iloc[-1][benchmark] #Get target val
   
   
   if plot_returns: #Plot SPY vs. Portfolio 
       plt.xlabel('Date')
       plt.xticks(rotation=45)
       plt.ylabel('Cumulative return')
       plt.title('Daily portfolio value and ' + benchmark[0])
       plt.grid()
       plt.plot(cumulative_returns['PORT'])
       plt.plot(cumulative_returns[benchmark])
       plt.legend(['Portfolio', benchmark[0]])
       plt.show()
       
    
   if display_output:
        print("Portfolio cumulative return: ", cumulative_return_port, "\ncumulative return benchmark: ", cumulative_return_bench)
        print("Average daily return: ", average_daily_return_port, "\naverage daily return portfolio: ", average_daily_return_bench)
        print("Daily return standard deviation: ", stdev_daily_return_port, "\ndaily return standard deviation benchmark: ", stdev_daily_return_bench)
        print("Sharpe ratio ", sharpe_ratio_port, "\nsharpe ratio benchmark: ", sharpe_ratio_bench)
        print("End value: ", end_value_port, "\nend value bench: ", end_value_bench)
       
   return end_value_port
    

if __name__ == '__main__':
    pd.set_option('display.float_format', lambda x: '%.20f' % x) 
    data = assess_strategy(trade_file = "./trades/leverage_checker.csv", starting_value = 1000000, fixed_cost = 0, floating_cost = 0)
    print(assess_portfolio(data, plot_returns=(True)))
    

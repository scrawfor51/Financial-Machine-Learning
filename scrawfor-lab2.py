# -*- coding: utf-8 -*-
"""
Stephen Crawford

Project 2: Assess a Stock Portfolio
Financial Machine Learning 
3/1/2022
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
             df = pd.read_csv(os.path.join(data_path, file), index_col='Date', parse_dates=True)
             df = df.loc[start_date : end_date]
             queried_data[file[:file.find('.')]] = df[column_name]
             
    return(queried_data)


"""
Main function to analyze a portfolio's performance.
Takes in a set of tickers, an list of corresponding allocations, and a date range and calculates common metrics.

@param start_date: The starting date for the range of time we are analyzing
@param end_date: The ending date for the range of time we are analyzing
@param symbols: A list of the tickers included in the portfolio 
@param allocations: A list of percentages as decimals of the starting budget's allocation to the tickers in the symbols param
@param starting_value: How much cash was initially invested in the portfolio
@param risk_free_rate: The expected level of returns for a 'risk-free' investment over a period of time
@param sample_freq: The sample frequency per year we are interested in standardizing to (accounts for inherent volatility in morre commonly traded stocks)
@param plot_returns: Whether we want to see a plot of SPY vs. the portfolio over the time period
@return cumulative_return: Zero-based value of the portfolio at the end date
@return average_daily_return: Zero-based average of daily portfolio returns
@return stdev_daily_return: Standard deviation of daily portfolio returns
@return sharpe_ratio: Annualized value indicating returns vs. risk
@return end_value: Dollar total of portfolio at end date
"""
def assess_portfolio (start_date, end_date, symbols, allocations,
                      starting_value=1000000, risk_free_rate=0.0,
                      sample_freq=252, plot_returns=False):
    
   data = get_data(start_date, end_date, symbols)
   data = data/data.iloc[0]
   data.iloc[:, 1:] = data.iloc[:, 1:] * allocations * starting_value
   data['PORT'] = data.drop('SPY', axis=1).sum(axis=1)
   
   cumulative_returns = (data/data.iloc[0] - 1)  # Get cumulative returns
   cumulative_return = cumulative_returns.iloc[-1]['PORT']
   average_daily_return = ((data/data.shift()).mean(axis=0) - 1)['PORT']# Get average daily returns 
   stdev_daily_return = (data/data.shift()).std(axis=0)['PORT']# Get standard devaition of daily returns 
   sharpe_ratio = ((data/data.shift() - 1 - risk_free_rate)/stdev_daily_return).mean() # calculate sharpe ratio
   sharpe_ratio = math.sqrt(sample_freq) * sharpe_ratio #Annualize sharpe ratio
   sharpe_ratio = sharpe_ratio['PORT'] # Get sharpe val
   end_value = data.iloc[-1]['PORT'] #Get target val
   
   if plot_returns: #Plot SPY vs. Portfolio 
       plt.xlabel('Date')
       plt.ylabel('Cumulative return')
       plt.title('Daily portfolio value and SPY')
       plt.grid()
       plt.plot(cumulative_returns['PORT'])
       plt.plot(cumulative_returns['SPY'])
       plt.legend(['Portfolio', 'SPY'])
       plt.show()

   return cumulative_return, average_daily_return, stdev_daily_return, sharpe_ratio, end_value
    
             
             
    
if __name__ == '__main__':
    
    pd.set_option('display.float_format', lambda x: '%.20f' % x) # See output as 9-significant-digit floats
 
    #Test case 1
    print(assess_portfolio('2010-01-01', '2010-12-31', ['GOOG', 'AAPL', 'GLD', 'XOM'], [0.2, 0.3, 0.4, 0.1], plot_returns=(True)))
    #Test case 2
    print(assess_portfolio('2015-06-30', '2015-12-31', ['MSFT', 'HPQ', 'IBM', 'AMZN'], [0.1, 0.1, 0.4, 0.4], plot_returns=(True), starting_value=10000, risk_free_rate=0.0022907680801))
    #Test case 3
    print(assess_portfolio('2020-01-01', '2020-06-30', ['NFLX', 'AMZN','XOM', 'PTON'], [0.0, 0.35, 0.35, 0.3], plot_returns=(True),  starting_value=500000, sample_freq=52))
    

    
    
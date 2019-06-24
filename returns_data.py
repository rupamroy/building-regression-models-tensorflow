import pandas as pd
import numpy as np


def read_goog_sp500_dataframe():
    googFile = 'stock_data/GOOG.csv'
    spFile = 'stock_data/SP_500.csv'


    goog = pd.read_csv(googFile, sep=",", usecols=[0,5], names=['Date', 'Goog'], header=0) # returns pandas dataframe object
    sp = pd.read_csv(spFile, sep=",", usecols=[0,5], names=['Date', 'SP500'], header=0)

    goog['SP500'] = sp['SP500'] # will merge the SP500 column from the sp data to the goog dataframe

    # the date object is a strinmg, convert that to date
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')

    # sort the data frame

    goog = goog.sort_values(['Date'], ascending=[True])

    pd.set_option('display.max_rows', 10)



    # calculate the percentage changes for the non Date type colums
    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]]\
        .pct_change()

    return returns


# Returns a tuple with 2 fields , the returns for Google and S&P 500
# Each of th returns are 1-D array
def read_goog_sp500_data():
    
    returns = read_goog_sp500_dataframe()

    # print 10 rows in the dataframe

    #print(goog)

    #print(returns)

    #filter out the first row which does not have any value for returns
    xData = np.array(returns["SP500"])[1:]
    yData = np.array(returns["Goog"])[1:]

    return(xData, yData)

def read_google_sp500_logistic_data():
    returns = read_goog_sp500_dataframe()

    returns["Intercept"] = 1

    xData = np.array(returns[["SP500", 'Intercept']][1:-1])

    yData = (returns["Goog"] > 0)[1:-1]

    return (xData, yData)
    


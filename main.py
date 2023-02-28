# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:55:03 2023
®
Hedge v5 - w/ buy signal, buy at open of next day
"""

"""
To do
- fix start date issue with beginning and end
- have a toggle for include shorting as well 
- have a commission (eg 0.1%)
- toggle for logarithmic charts
- output a date where the maximum drawdown occurs
- output the number of closed trades, average duration of trade, average PnL per trade

"""


# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yfinance as yf
import datetime as dt
import copy
from tabulate import tabulate
import os
port = int(os.environ.get('PORT', 5000))

import warnings
warnings.filterwarnings("ignore")

symbol = "nvda"
short = 1  # short EMA/SMA period (days)
long = 200  # long EMA/SMA period (days)
ind = "SMA"  # choose EMA or SMA


# Define dictionaries
ohlc_data = {}

# Download historical data for tickers
start_years_ago = 30
end_years_ago = 0    #set as zero for today
start = dt.datetime.today() - dt.timedelta(days=365 * start_years_ago)
end = dt.datetime.today() - dt.timedelta(days=365 * end_years_ago)


# Define functions
def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["Returns"]).cumprod()
    n = len(df) / 252
    CAGR = (df["cum_return"].tolist()[-1]) ** (1 / n) - 1
    return CAGR

def total_return_multiple(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    total_return_multiple = df["total return multiple"] = (1 + df["Returns"]).cumprod()
    return total_return_multiple[-1]


def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["Returns"].std() * np.sqrt(252)
    return vol


def sharpe(DF, rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf) / volatility(df)
    return sr


def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["Returns"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


tickers = [symbol]

# Other ticker: '^IXIC','ARKK','GOOG','QQQ',

for ticker in tickers:
    try:
        ohlc_data[ticker] = yf.download(ticker, start, end)
    except Exception as e:
        print("Failed to download data for ticker {}: {}".format(ticker, str(e)))

ticker_signal = symbol
ticker_strat = symbol

# Calculating SMA/EMA signal
ohlc_dict = copy.deepcopy(ohlc_data)  # copy original data
cl_price = pd.DataFrame()

print('Calculating SMA and EMA for ', ticker_signal)
ohlc_dict[ticker_signal]['Short SMA'] = ohlc_dict[ticker_signal]['Adj Close'].rolling(window=short).mean()
ohlc_dict[ticker_signal]['Long SMA'] = ohlc_dict[ticker_signal]['Adj Close'].rolling(window=long).mean()
ohlc_dict[ticker_signal]['Short EMA'] = ohlc_dict[ticker_signal]['Adj Close'].ewm(span=short, adjust=False).mean()
ohlc_dict[ticker_signal]['Long EMA'] = ohlc_dict[ticker_signal]['Adj Close'].ewm(span=long, adjust=False).mean()
cl_price[ticker_signal] = ohlc_dict[ticker_signal]['Adj Close']

print('Calculating Buy/Sell signal for ', ticker_signal)
ohlc_dict[ticker_signal]['Signal'] = 0.0
ohlc_dict[ticker_signal]['Signal'] = np.where(
    ohlc_dict[ticker_signal]['Short {}'.format(ind)] > ohlc_dict[ticker_signal]['Long {}'.format(ind)], 1.0, 0.0)
ohlc_dict[ticker_signal]['Signal'] = ohlc_dict[ticker_signal]['Signal'].shift(1)
ohlc_dict[ticker_signal]['Position'] = ohlc_dict[ticker_signal]['Signal'].diff()

# Calculate returns with long strategy and ticker signal
df_temp = pd.concat(ohlc_dict, axis=1)
strat_returns = df_temp.xs('Adj Close', axis=1, level=1)
strat_returns['Position'] = df_temp.xs('Position', axis=1, level=1)
strat_returns = strat_returns.copy()
strat_returns['Signal'] = df_temp.xs('Signal', axis=1, level=1)
strat_returns['Returns'] = 0

for i in range(1, len(strat_returns)):
    if strat_returns['Signal'].iloc[i] == 1:
        strat_returns['Returns'].iloc[i] = (
                    (strat_returns[ticker_strat].iloc[i] / strat_returns[ticker_strat].iloc[i - 1]) - 1)
    else:
        strat_returns['Returns'].iloc[i] = 0

strat_returns['All Returns'] = (strat_returns[ticker_strat].pct_change())

# Calculating long-only strategy KPIs without signal
strategy_df_2 = pd.DataFrame()
strategy_df_2["Returns"] = strat_returns["All Returns"]
strategy_df_2["Returns"] = strategy_df_2.mean(axis=1)
strategy_df_2["cum_return"] = (1 + strategy_df_2["Returns"]).cumprod()

# Calculating long-only strategy KPIs with signal
strategy_df = pd.DataFrame()
strategy_df["Returns"] = strat_returns["Returns"]
strategy_df["Returns"] = strategy_df.mean(axis=1)

def main():
    # Charts
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    #ax[0].set_title('Long-only strategy: {a}'.format(a=ticker_strat))
    ax[0].set_title('Crossover signal: {a} {b}/{c} {d} '.format(a=ticker_signal, b=short, c=long, d=ind))
    ax[1].set_title('Cumulative return')

    ax[0].grid()
    ax[1].grid()
    #ax[2].grid()

    # Chart 1
    #ax[0].plot(strat_returns[ticker_strat], color='black')

    # Chart 2
    ax[0].plot(ohlc_dict[ticker_signal]['Adj Close'], color='black', label='Adj Close')
    ax[0].plot(ohlc_dict[ticker_signal]['Short {}'.format(ind)], color='black', label='Short {}'.format(ind))
    ax[0].plot(ohlc_dict[ticker_signal]['Long {}'.format(ind)], color='g', label='Long {}'.format(ind))

    buys = ohlc_dict[ticker_signal][ohlc_dict[ticker_signal]['Position'] == 1].index
    sells = ohlc_dict[ticker_signal][ohlc_dict[ticker_signal]['Position'] == -1].index
    ax[0].plot_date(buys, ohlc_dict[ticker_signal]['Short {}'.format(ind)][ohlc_dict[ticker_signal]['Position'] == 1], \
                    '^', markersize=5, color='g', label='buy')
    ax[0].plot_date(sells, ohlc_dict[ticker_signal]['Short {}'.format(ind)][ohlc_dict[ticker_signal]['Position'] == -1], \
                    'v', markersize=5, color='r', label='sell')
    ax[0].legend()

    # Chart 3
    strategy_df_2["cum_return"] = (1 + strategy_df_2["Returns"]).cumprod()
    strategy_df_2['Position'] = strat_returns['Position']
    ax[1].plot(strategy_df_2["cum_return"])
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Chart 4
    strategy_df["cum_return"] = (1 + strategy_df["Returns"]).cumprod()
    strategy_df['Position'] = strat_returns['Position']
    ax[1].plot(strategy_df["cum_return"],color='green')
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Print output: KPIs table
    table = (tabulate([['CAGR', "{:.2%}".format(CAGR(strategy_df)), "{:.2%}".format(CAGR(strategy_df_2))],
                       ['Sharpe ratio', "{:.2f}".format(sharpe(strategy_df, 0.025)),
                        "{:.2f}".format(sharpe(strategy_df_2, 0.025))],
                       ['Max Drawdown', "{:.2%}".format(max_dd(strategy_df)), "{:.2%}".format(max_dd(strategy_df_2))],
                       ['Total return multiple', "{:.2%}".format(total_return_multiple(strategy_df)), "{:.2%}".format(total_return_multiple(strategy_df_2))]],
                      headers=['KPI', 'Long-only strat w/ signal', 'Long-only strat w/o signal'],
                      tablefmt='orgtbl'))
    print(table)

    plt.show()


if __name__ == "__main__":
    main()

    """
    # Chart buy/sell signal
    for c in ax:
        for i in ticker_signal:
            xs = np.linspace(1, 21, 200)
            buys = ohlc_dict[i][ohlc_dict[i]['Position'] == 1].index
            sells = ohlc_dict[i][ohlc_dict[i]['Position'] == -1].index
            c.vlines([buys], ymin = 0, ymax = len(xs), linestyles='dashed', colors='green')
            c.vlines([sells], ymin = 0, ymax = len(xs), linestyles='dashed', colors='red')

    # Table in subplots 
    collabel=("CAGR", "Sharpe Ratio", "Max Drawdown")
    rowdata=((["{:.2%}".format(CAGR(strategy_df))]),["{:.2f}".format(sharpe(strategy_df,0.025))], 
             ["{:.2%}".format(max_dd(strategy_df))])
    #rowdata = (CAGR(strategy_df), sharpe(strategy_df,0.025), max_dd(strategy_df))
    table = table(cellText=rowdata,rowLabels=collabel,loc='center', colWidths = [0.3])
    table.set_fontsize(10)
    table.scale(1.2, 1.2)


    """

    """
    stock price crossing a specific moving averages that was optimised for something… 
    max return and/or minimise drawdown with long 1x ARKK as a stand in for my strategy

"""






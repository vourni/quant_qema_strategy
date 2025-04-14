import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import optuna
import os
import datetime as dt
from signals import apply_signals

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_rows', 500)

# defining keys
API_KEY = 'PK78EB0ZSWAZF9XHSCIW'
SECRET_API_KEY = '8uCbF9z8zuePB1MwhzFQj7uHHLnhg2Uo1YJru0EW'
BASE_URL = 'https://paper-api.alpaca.markets'

# initializing api
api = tradeapi.REST(API_KEY, SECRET_API_KEY, BASE_URL, api_version='v2')
account = api.get_account()

class Ticker:
    def __init__(self, ticker, start, end, initial_balance):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.initial_balance = initial_balance

        self.train_sharpe = None
        self.test_pf = None
        self.test_sharpe = None
        self.test_return = None

        self.qema_period = None
        self.ema_period = None

        self.train_data = None
        self.test_data = None

        self.best_data = None
        

        # setting filename
        filename = f"{self.ticker}_{self.start}_{self.end}_5min.csv"

        # checking if file exists
        if os.path.exists(filename):
            # getting csv
            print(f"📁 Loading cached data from {filename}")
            self.data = pd.read_csv(filename, index_col=0, parse_dates=True)
            self.data.index = pd.to_datetime(self.data.index, utc=True).tz_convert('America/New_York')
         
        else: 
            # donwloading data
            self.data = api.get_bars(self.ticker, timeframe='5min', limit=10000000000, start=self.start, end=self.end, adjustment='split').df

            # setting timestamp
            self.data['timestamp'] = self.data.index 
            self.data['timestamp'] = self.data['timestamp'].dt.tz_convert('America/New_York')
            self.data.set_index(self.data['timestamp'], inplace=True)
            self.data.drop('timestamp', axis=1, inplace=True)

            # calculating log returns
            self.data['log_return'] = np.log(self.data['close']).diff().shift(-1).fillna(0)

            self.data.to_csv(filename)


    def calculate_returns(self, data):
        # calculating strategy returns and cumsum
        data['strategy_return'] = data['log_return'] * data['position']
        data['cumulative_strategy_returns'] = data['strategy_return'].cumsum()

        # appling to balance
        data['balance'] = self.initial_balance * np.exp(data['cumulative_strategy_returns'])

        # applying to basis
        data['basis_cumsum'] = data['log_return'].cumsum()
        data['basis'] = self.initial_balance * np.exp(data['basis_cumsum'])

        return data


    def objective(self, trial):
        # setting vars
        qema_period = trial.suggest_int("qema", 20, 100)
        ema_period = trial.suggest_int("ema", 100, 300)

        # create train and validation set
        train_size = int(len(self.train_data) * 0.8)
        train_set = self.train_data.iloc[:train_size].copy()
        validation_set = self.train_data.iloc[train_size:].copy()

        # apply indicators, signals, and returns
        for dataset in [train_set, validation_set]:
            dataset = apply_signals(dataset, qema_period, ema_period)
            dataset = self.calculate_returns(dataset)

        # get returns
        train_returns = train_set['strategy_return']
        validation_returns = validation_set['strategy_return']

        # get profit factor
        train_pf = train_returns[train_returns > 0].sum() / train_returns[train_returns < 0].abs().sum() if train_returns[train_returns < 0].abs().sum() != 0 else 0
        validation_pf = validation_returns[validation_returns > 0].sum() / validation_returns[validation_returns < 0].abs().sum() if validation_returns[validation_returns < 0].abs().sum() != 0 else 0

        # other factors
        consistency = 1 - min(abs(train_pf - validation_pf) / max(train_pf, validation_pf), 0.5)
        
        trade_count = (train_set['position'].diff() != 0).sum()
        ideal_trades = len(train_set) * 0.005  
        trade_penalty = max(0, 1 - min(abs(trade_count - ideal_trades) / ideal_trades, 0.5))

        # scoring
        objective_score = (0.4 * validation_pf + 
                    0.3 * train_pf + 
                    0.2 * consistency + 
                    0.1 * trade_penalty)
    
        return objective_score
    

    def optimize(self, data):
        # setting train and test data
        split_index = int(len(data) * 0.7)
        self.train_data = data.iloc[:split_index].copy()
        self.test_data = data.iloc[split_index:].copy()

        # running bayesian optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, n_jobs=-1)
        
        # setting class level vars
        self.ema_period = study.best_params['ema']
        self.qema_period = study.best_params['qema']


    def get_test_data(self):
        # getting best data
        data = self.test_data.copy()
        data = apply_signals(data, self.qema_period, self.ema_period)
        data = self.calculate_returns(data)

        # getting returns
        r = data['strategy_return']

        # adjusting df
        self.best_data = data

        # getting sharpe
        self.test_pf = r[r>0].sum() / r[r<0].abs().sum()
        self.test_sharpe = (r.mean() / r.std()) * np.sqrt(48384)
        self.test_return = ((self.initial_balance * np.exp(r.cumsum().iloc[-1])) / self.initial_balance - 1) * 100
    

    def display_results(self):
        # displating relevant stats
        print(self.best_data)
        print('# Best:')
        print('# QEMA Lookback:', self.qema_period)
        print('# 1H EMA Period:', self.ema_period)
        print('# Test Sharpe Ratio:', round(self.test_sharpe, 2))
        print('# Test Profit Factor:', round(self.test_pf, 2))
        print('# Test Return:', round(self.test_return, 2), '%')


    def permute_returns(self):
        data = self.data.copy()
        shuffled = data['log_return'].sample(frac=1.0, replace=False).reset_index(drop=True)
        data['log_return'] = shuffled.values
        data['price_sim'] = data['close'].iloc[0] * np.exp(data['log_return'].cumsum())
        data['close'] = data['price_sim']
        return data


    def test_perm_data(self, n):
        # Initializing lists
        sharpes = []
        pfs = []
        returns = []

        # Permuting for n trials
        for i in range(n):
            print(f'Trial {i+1}')

            # getting data and optimizing on data
            data = self.permute_returns()
            self.optimize(data)
            self.get_test_data()

            # appending data
            sharpes.append(self.test_sharpe)
            pfs.append(self.test_pf)
            returns.append(self.test_return)

        # creating datafame
        df = pd.DataFrame({'Sharpes' : sharpes, 'PFs': pfs, 'Returns': returns})
        return df

    

if __name__ == '__main__':
    yesterday = dt.date.today() - dt.timedelta(days=2) # yesterdays date
    tqqq = Ticker('TQQQ', start='2024-01-01', end=yesterday, initial_balance=1000) # initializing class for stock

    tqqq.optimize(tqqq.data)
    tqqq.get_test_data()
    tqqq.display_results()
    real_pf = tqqq.test_pf

    #n = 5
    #df = tqqq.test_perm_data(n)

    #print(df, real_pf)

    #counter = 0
    #for index, row in df.iterrows():
    #    if row['PFs'] > real_pf:
    #        counter += 1

    #print('Real results are better than', round((1-(counter/n)) * 100, 2),'%' ,'of permuted results')
    #plt.hist(df[['PFs']])
    #plt.vlines(real_pf, ymin=0, ymax=len(df)/2, color='red')
    #plt.show()

# Best:
# Train Sharpe Ratio: 2.36
# Test Sharpe Ratio: 2.33
# Test Profit Factor: 1.13
# Test Return: 303.03 %
# Best QEMA Lookback: 106
# Best 1H EMA Period: 113

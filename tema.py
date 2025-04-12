import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import optuna
import ta.trend
import ta.volatility
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
        self.test_pf = None
        self.train_sharpe = None
        self.test_sharpe = None
        self.qema_period = None
        self.fast_period = None
        self.slow_period = None
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
            self.data = api.get_bars(self.ticker, timeframe='5min', limit=10000000000, start=self.start, end=self.end).df

            # setting timestamp
            self.data['timestamp'] = self.data.index 
            self.data['timestamp'] = self.data['timestamp'].dt.tz_convert('America/New_York')
            self.data.set_index(self.data['timestamp'], inplace=True)
            self.data.drop('timestamp', axis=1, inplace=True)

            self.data.to_csv(filename)

    def apply_indicators(self, data, period, fast_period, slow_period):
        # finding high/low/close average
        data['hlc3'] = round((data['high'] + data['low'] + data['close']) / 3, 2)

        # calculating emas
        data['ema_1'] = data['hlc3'].ewm(span=period, adjust=False).mean()
        data['ema_2'] = data['ema_1'].ewm(span=period, adjust=False).mean()
        data['ema_3'] = data['ema_2'].ewm(span=period, adjust=False).mean()

        # calculating fast and slow emas
        data['fast_ema'] = data['hlc3'].ewm(span=fast_period, adjust=False).mean()
        data['slow_ema'] = data['hlc3'].ewm(span=slow_period, adjust=False).mean()

        # calculating tema/qema
        data['TEMA'] = (3 * data['ema_1']) - (3 * data['ema_2']) + data['ema_3']
        data['QEMA'] = round(data['TEMA'].ewm(span=3, adjust=False).mean(), 2)


        # reampling dataframe for 1 hour aggregates
        data_1h = data.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        # 200 day ema for regime detection
        data_1h['ema_200'] = data_1h['close'].ewm(span=200, adjust=False).mean()

        # adx and atr for volatility
        data_1h['adx'] = ta.trend.adx(data_1h['high'], data_1h['low'], data_1h['close'], window=14)
        data_1h['atr'] = ta.volatility.average_true_range(data_1h['high'], data_1h['low'], data_1h['close'], window=14)

        # applying regime labels
        data_1h['regime'] = np.where(data_1h['adx'] > 20, np.where(data_1h['close'] >= data_1h['ema_200'], 'uptrend', 'downtrend'), 'neutral')
        data_1h['volatility_regime'] = np.where(data_1h['atr'] > data_1h['atr'].rolling(100).mean(), 'high_vol', 'low_vol')

        # resampling to 5 mins
        data['regime'] = data_1h['regime'].reindex(data.index, method='ffill')
        data['volatility_regime'] = data_1h['volatility_regime'].reindex(data.index, method='ffill')

        # removing columns
        data.drop(['ema_1', 'ema_2', 'ema_3', 'TEMA'], axis=1, inplace=True)

        return data
    

    def calculate_returns(self, data):
        # calculating log returns
        data['log_return'] = np.log(data['close']).diff().shift(-1).fillna(0)

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
        # setting train and test data
        split_index = int(len(self.data) * 0.7)
        self.train_data = self.data.iloc[:split_index].copy()
        self.test_data = self.data.iloc[split_index:].copy()
        
        # setting vars
        period = trial.suggest_int("qema", 10, 100)
        fast = trial.suggest_int("fast", 10, 60)
        slow = trial.suggest_int("slow", 60, 200)

        # checking for incalid combos
        if fast >= slow:
            raise optuna.exceptions.TrialPruned()

        # apply indicators, signals, and returns
        temp = self.train_data.copy()
        temp = self.apply_indicators(temp, period, fast, slow)
        temp = apply_signals(temp)
        temp = self.calculate_returns(temp)

        # get returns
        r = temp['strategy_return']

        # get profit factor
        sharpe = (r.mean() / r.std()) * np.sqrt(48384)
        return sharpe
    

    def optimize(self):
        # running bayesian optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, n_jobs=-1)
        
        # setting class level vars
        self.train_sharpe = study.best_value
        self.fast_period = study.best_params['fast']
        self.slow_period = study.best_params['slow']
        self.qema_period = study.best_params['qema']


    def get_test_data(self):
        # getting best data
        data = self.test_data.copy()
        data = self.apply_indicators(data, self.qema_period, self.fast_period, self.slow_period)
        data = apply_signals(data)
        data = self.calculate_returns(data)

        # getting returns
        r = data['strategy_return']

        # adjusting df
        data.drop(['high', 'low', 'trade_count', 'open', 'volume', 'vwap', 'hlc3', 'position'], axis=1, inplace=True)
        self.best_data = data

        # getting sharpe
        self.test_pf = r[r>0].sum() / r[r<0].abs().sum()
        self.test_sharpe = (r.mean() / r.std()) * np.sqrt(48384)
    

    def display_results(self):
        # displating relevant stats
        print(self.best_data)
        print('# Best:')
        print('# Train Sharpe Ratio:', self.train_sharpe)
        print('# Test Sharpe Ratio:', self.test_sharpe)
        print('# Test Profit Factor:', self.test_pf)
        print('# Best QEMA Lookback:', self.qema_period)
        print('# Best Fast EMA Period:', self.fast_period)
        print('# Best Slow EMA Period:', self.slow_period)


if __name__ == '__main__':
    yesterday = dt.date.today() - dt.timedelta(days=1) # yesterdays date
    tqqq = Ticker('TQQQ', start='2020-01-01', end=yesterday, initial_balance=1000) # initializing class for stock

    tqqq.optimize() # optimizing parameters
    tqqq.get_test_data() # get test data
    tqqq.display_results() # printing data
    plt.plot(tqqq.best_data['balance'].values)
    plt.plot(tqqq.best_data['basis'].values)
    plt.show()

# Best:
# Train Sharpe Ratio: 1.2884923826643435
# Test Sharpe Ratio: 1.6017092157729809
# Test Profit Factor: 1.079949908112905
# Best QEMA Lookback: 25
# Best Fast EMA Period: 10
# Best Slow EMA Period: 129
    

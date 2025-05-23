import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna
import os
import datetime as dt

from secret import API_KEY, SECRET_API_KEY, BASE_URL

from generate_signals import signals

# initializing api
api = tradeapi.REST(API_KEY, SECRET_API_KEY, BASE_URL, api_version='v2')

class Ticker:
    def __init__(self, ticker, start, end, timeframe, initial_balance):
        # intizializing variables
        self.ticker = ticker
        self.start = start
        self.end = end
        self.timeframe = timeframe
        self.initial_balance = initial_balance

        # calling _load_data
        self._load_data()
        

    def _load_data(self):
        # setting filename
        filename = os.path.join('price_data', f"{self.ticker}_{self.start}_{self.end}_{self.timeframe}.csv")

        # checking if file exists
        if os.path.exists(filename):
            # getting csv
            print(f"Loading cached data from {filename}")
            self.data = pd.read_csv(filename, index_col=0, parse_dates=True)
            self.data.index = pd.to_datetime(self.data.index, utc=True).tz_convert('America/New_York')
         
        else: 
            # donwloading data
            self.data = api.get_bars(self.ticker, timeframe=self.timeframe, limit=10000000000, start=self.start, end=self.end, adjustment='split').df

            # setting timestamp
            self.data['timestamp'] = self.data.index 
            self.data['timestamp'] = self.data['timestamp'].dt.tz_convert('America/New_York')
            self.data.set_index(self.data['timestamp'], inplace=True)
            self.data.drop('timestamp', axis=1, inplace=True)

            # calculating log returns
            self.data['log_return'] = np.log(self.data['close']).diff().fillna(0)

            # saving to csv
            self.data.to_csv(filename)


    def apply_signals(self, data, period_q, period_d, bars):
        # applying signals
        data = signals(data, period_q, period_d, bars)
        return data
        

    def apply_stop_loss(self, data, atr_window=14, atr_multiplier=2, cooldown=5):
        # copying data
        data = data.copy()

        # calculating average true range
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr'] = tr.rolling(window=atr_window, min_periods=1).mean()

        # grouping by trades
        position_change = (data['position'] != data['position'].shift(1))
        data['entry_price'] = data['close'].where(position_change).ffill()
        data['entry_atr'] = data['atr'].shift(1).where(position_change).ffill()

        # atr based threshholds
        data['atr_pct'] = atr_multiplier * data['entry_atr'] / data['entry_price']
        data['atr_pct'] = data['atr_pct'].clip(upper=0.99)
        data['log_sl_long'] = np.log(1 - data['atr_pct'])
        data['log_sl_short'] = np.log(1 + data['atr_pct'])

        # calculating draw downs
        data['drawdown_long'] = np.log(data['close'] / data['entry_price'])
        data['drawdown_short'] = np.log(data['entry_price'] / data['close'])

        # triggering stop loss
        long_exit = (data['drawdown_long'] < data['log_sl_long']) & (data['position'] == 1)
        short_exit = (data['drawdown_short'] < data['log_sl_short']) & (data['position'] == -1)
        stop_loss_triggered = long_exit | short_exit

        # setting position to 0 if stop loss triggered
        data.loc[stop_loss_triggered, 'position'] = 0

        # setting cool-down
        new_position = []
        cooldown_counter = 0
        for i in range(len(data)):
            # checking conditions
            if stop_loss_triggered.iloc[i]:
                # setting counter
                cooldown_counter = cooldown
                # appending new positions
                new_position.append(0)
            elif cooldown_counter > 0:
                new_position.append(0)
                # decreasing counter
                cooldown_counter -= 1
            else:
                new_position.append(data['position'].iloc[i])

        # updating positions after cooldown
        data['position'] = new_position

        # cleaning
        data.drop(columns=['atr', 'entry_price', 'entry_atr', 'atr_pct',
                            'log_sl_long', 'log_sl_short',
                            'drawdown_long', 'drawdown_short'], inplace=True)

        return data
        

    def calculate_returns(self, data):
        # transaction cost
        data['position_change'] = data['position'].diff().fillna(0)
        data['transaction_cost'] = abs(data['position_change']) * 0.001

        # calculating strategy returns and balance
        data['strategy_return'] = data['log_return'] * data['position'] - data['transaction_cost']
        data['cumulative_strategy_returns'] = data['strategy_return'].cumsum()
        data['balance'] = self.initial_balance * np.exp(data['cumulative_strategy_returns'])

        # applying to basis
        data['basis_cumsum'] = data['log_return'].cumsum()
        data['basis'] = self.initial_balance * np.exp(data['basis_cumsum'])

        return data
    

    def optimize(self, data, n_trials=30):
        # objective function
        def objective(trial):
                # setting params
                q = trial.suggest_int('qema_period', 80, 150)
                d = trial.suggest_int('deriv_period', 1, 5)
                b = trial.suggest_int('bars', 1, 3)
                aw = trial.suggest_int('atr_window', 14, 24)
                am = trial.suggest_int('atr_multiplier', 10, 20)
                c = trial.suggest_int('cooldown', 3, 7)

                # setting objective
                _,sharpe,pf,_ = self.backtest(data.copy(), q, d, b, aw, am/10, c)
                return sharpe + pf

            
        # executing study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # return params
        best = study.best_params
        return best['qema_period'], best['deriv_period'], best['bars'], best['atr_window'], best['atr_multiplier'] / 10, best['cooldown']
    

    def backtest(self, data, period_q, period_d, bars, atr_window, atr_multiplier, cooldown):
        # getting signals and returns
        data = self.apply_signals(data, period_q, period_d, bars)
        data = self.apply_stop_loss(data, atr_window, atr_multiplier, cooldown)
        data = self.calculate_returns(data)

        # setting strat returns
        r = data['strategy_return']

        # calculating pf and sharpe
        sharpe = r.mean() / r.std() * np.sqrt(252*16)
        pf = r[r>0].sum() / r[r<0].abs().sum()

        # getting signal hit rate
        data_positions = data[data['position'] != 0]
        return_direction = np.sign(data_positions['log_return'])
        signal_direction = np.sign(data_positions['position'])
        correct_direction = (return_direction == signal_direction)
        hit_rate = correct_direction.mean()

        # cleaning
        data = data[['close', 'log_return', 'position', 'strategy_return', 'balance', 'basis']]

        return data, sharpe, pf, hit_rate
    

    def permute_data(self):
        # copying data
        data = self.data
        result = data.copy()
        
        # shuffling log returns
        result['log_return'] = np.random.permutation(data['log_return'].values)

        # simulating price
        result['close'] = data['close'].iloc[0] * np.exp(result['log_return'].cumsum())

        # simuulating other factors
        ratio = result['close'] / data['close']
        for col in ['high', 'low', 'open', 'volume']:
            result[col] = data[col] * ratio

        # cleaning dataframe
        result.drop(columns=[col for col in data.columns if col not in ['close', 'log_return', 'open', 'high', 'volume', 'low']], inplace=True)
        result.dropna(inplace=True)

        return result
    

    def permute_data_blocks(self, block_size=48):
        data = self.data.copy()
        log_returns = data['log_return'].dropna().values

        # Divide log returns into blocks
        n_blocks = len(log_returns) // block_size
        blocks = [log_returns[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]

        # Shuffle blocks
        np.random.shuffle(blocks)
        bootstrapped_log_return = np.concatenate(blocks)

        # Rebuild synthetic close price
        close = [data['close'].iloc[0]]
        for r in bootstrapped_log_return:
            close.append(close[-1] * np.exp(r))
        close = pd.Series(close[1:], index=data.index[:len(bootstrapped_log_return)])

        # Rebuild DataFrame
        result = pd.DataFrame(index=close.index)
        result['close'] = close
        result['log_return'] = np.log(close / close.shift(1)).fillna(0)

        # Rescale OHLCV to match new close trajectory
        ratio = result['close'] / data['close'].iloc[:len(result)]
        for col in ['high', 'low', 'open', 'volume']:
            result[col] = data[col].iloc[:len(result)].values * ratio.values

        return result.dropna()


    def run_permutation_test(self, n):
        # initialzing vectors
        sharpes = []
        pfs = []
        hrs = []

        # looping through trials
        for i in range(n):
            # permuting data
            temp = self.permute_data_blocks()

            # optimizing
            q,d,b,aw,am,c = self.optimize(temp)

            # getting data stats
            temp_data, sharpe, pf, hit_rate = self.backtest(temp, period_q=q, period_d=d, bars=b, atr_window=aw, atr_multiplier=am, cooldown=c)

            # apending lists
            sharpes.append(sharpe)
            pfs.append(pf)
            hrs.append(hit_rate)
            print(i)
        
        # creting df
        df = pd.DataFrame({
            'Sharpes' : sharpes,
            'Profit Factors' : pfs,
            'Hit Rates' : hrs,
        })

        return df
    

    def get_real_data(self, data, atr_window=14, atr_multiplier=2, cooldown=5):
        # optimizng real data
        q,d,b,aw,am,c = self.optimize(data)
        params = {
            'period_q':q,
            'period_d':d,
            'bars':b,
            'atr_window':aw,
            'atr_mult':am,
            'countdown':c
        }

        self.best_params = pd.DataFrame.from_dict([params])    

        real_dataframe, real_sharpe, real_pf, real_hr = self.backtest(data, period_q=q, period_d=d, bars=b, atr_window=aw, atr_multiplier=am, cooldown=c)

        # creating list
        real_data = [real_dataframe, real_sharpe, real_pf, real_hr]

        return real_data

    
    def run_results(self, perms, real_data):
        #unpacking list
        real_dataframe = real_data[0]
        real_sharpe = real_data[1]
        real_pf = real_data[2]
        real_hr = real_data[3]

        # getting percentiles
        sharpe_percentile = (perms['Sharpes'] < real_sharpe).mean() * 100
        pf_percentile = (perms['Profit Factors'] < real_pf).mean() * 100
        hr_percentile = (perms['Hit Rates'] < real_hr).mean() * 100

        # max drawdown
        cumulative = real_dataframe['balance']
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = drawdown.min()

        # simple return
        simple_return = np.exp(real_dataframe['strategy_return']) - 1
        cum_simple_return = (1 + simple_return).cumprod().iloc[-1] - 1

        # win rate and average win
        trade_return = real_dataframe['strategy_return'].where(real_dataframe['position'] != 0)
        win_rate = (trade_return > 0).sum() / trade_return.count()
        avg_win = trade_return[trade_return > 0].mean()
        avg_loss = trade_return[trade_return < 0].mean()

        # average trade length and number of trades
        df = real_dataframe.copy()
        df['position_change'] = df['position'] != df['position'].shift()
        df['trade_id'] = df['position_change'].cumsum()
        df['in_trade'] = df['position'] != 0
        df['trade_id'] = df['trade_id'].where(df['in_trade'])
        trade_lengths = df.groupby('trade_id').size()

        average_trade_length = trade_lengths.mean()
        num_trades = trade_lengths.count()

        # printing percentiles
        print('\n' + '=' * 33 + ' TEST RESULTS ' + '=' * 33 + '\n')

        print(f'Real Sharpe ({real_sharpe:.2f}) is better than {sharpe_percentile:.2f}% of permuted results')
        print(f'Real Profit Factor ({real_pf:.2f}) is better than {pf_percentile:.2f}% of permuted results')
        print(f'Real Hit Rate ({real_hr * 100:.2f}%) is better than {hr_percentile:.2f}% of permuted results')

        print('\n' + '-' * 80 + '\n')

        print(f'Max Drawdown: {round(max_drawdown * 100, 2)}%')
        print(f'Arithmetic Return: {round(cum_simple_return, 2)}x')

        print('\n' + '-' * 80 + '\n')

        print(f'Number of Trades: {num_trades}')
        print(f'Average Trade Length: {round(average_trade_length, 2)} hours')
        print(f'Win Rate: {round(win_rate * 100, 2)}%')
        print(f'Average Win: {round(avg_win * 100, 2)}% per hour of winning trade')
        print(f'Average Loss: {round(avg_loss * 100, 2)}% per hour of losing trade')
        

        print('\n' + '-' * 80 + '\n')

        print('Best Parameters:')
        print(self.best_params)

        print('\n' + '=' * 80 + '\n')

        # setting figure and plots
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        
        # sharpe ratio histogram
        axes[0].hist(perms['Sharpes'], bins=20, alpha=0.7)
        axes[0].axvline(real_sharpe, color='red', linestyle='dashed', linewidth=2, label=round(real_sharpe, 2))
        axes[0].legend()
        axes[0].set_title('Distribution of Sharpe Ratios')
        axes[0].set_xlabel('Sharpe Ratio')
        axes[0].set_ylabel('Frequency')
        
        # profit factor histogram
        axes[1].hist(perms['Profit Factors'], bins=20, alpha=0.7)
        axes[1].axvline(real_pf, color='red', linestyle='dashed', linewidth=2, label=round(real_pf, 2))
        axes[1].legend()
        axes[1].set_title('Distribution of Profit Factors')
        axes[1].set_xlabel('Profit Factor')
        axes[1].set_ylabel('Frequency')
        
        # hit rates histogram
        axes[2].hist(perms['Hit Rates'], bins=20, alpha=0.7)
        axes[2].axvline(real_hr, color='red', linestyle='dashed', linewidth=2, label=round(real_hr*100, 2))
        axes[2].legend()
        axes[2].set_title('Distribution of Hit Rates (%)')
        axes[2].set_xlabel('Hit Rate (%)')
        axes[2].set_ylabel('Frequency')

        # plotting balance and basis
        axes[3].plot(real_dataframe['basis'].values, color="red", label='Basis')
        axes[3].plot(real_dataframe['balance'].values, color="blue", label='Strategy')
        axes[3].legend()
        axes[3].set_title('Buy/Hold Vs. Strategy')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Dollars ($USD)')

        # saving and plotting
        plt.tight_layout()
        plt.savefig(f"strategy_plots_{self.ticker}.pdf", format="pdf", bbox_inches="tight")

        # saving csv 
        real_dataframe.to_csv('data.csv')


if __name__ == '__main__':

    # intializing class
    yesterday = dt.date.today() - dt.timedelta(days=2) # date two days ago
    TQQQ = Ticker('TQQQ', start='2016-01-01', end=yesterday, timeframe='1hour', initial_balance=1000) # initializing class for stock


    # getting real data 
    real_data = TQQQ.get_real_data(TQQQ.data)

    # running permutation test
    perms = TQQQ.run_permutation_test(n=0)

    # plotting data
    TQQQ.run_results(perms, real_data)

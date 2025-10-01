import numpy as np
import pandas as pd

def signals(data, qema_period,
                  deriv_period,
                  confirm_bars):
    
    #taking seris of 3 emas 
    ema1 = data['close'].ewm(span=qema_period, adjust=False).mean()
    ema2 = ema1.ewm(span=qema_period, adjust=False).mean()
    ema3 = ema2.ewm(span=qema_period, adjust=False).mean()

    # calculing tema and qema
    tema = 3*ema1 - 3*ema2 + ema3
    data['qema'] = tema.ewm(span=qema_period, adjust=False).mean()

    # qema bands
    stdev = data['qema'].rolling(20).std()
    data['qema_up'] = data['qema'] + 1.5 * stdev
    data['qema_low'] = data['qema'] - 1.5 * stdev

    # calculating 1st/2nd derivatives
    data['qema_d1'] = data['qema'].diff(deriv_period)
    data['qema_d2'] = data['qema_d1'].diff(deriv_period)

    # simple trend
    data['sma200'] = data['close'].rolling(200).mean()


    # setting position and shifting forward for look-ahead bias
    data['signal'] = np.where((data['close'] > data['qema_up']) & 
                                (data['qema_d2'] > 0) &
                                (data['close'] > data['sma200']),
                                1,
                        np.where((data['close'] < data['qema_low']) & 
                                (data['qema_d2'] < 0) &
                                (data['close'] < data['sma200']),
                                -1,
                                0))
    
    rolling_sum = data['signal'].rolling(confirm_bars, min_periods=1).sum()
    is_consistent = (rolling_sum == data['signal'] * confirm_bars)
    data['position'] = np.where(is_consistent, data['signal'], 0)
    
    data['position'] = data['position'].shift(1)

    # cleaning
    data.drop(['qema', 'qema_d1', 'qema_d2'], axis=1, inplace=True)
    data.dropna(inplace=True)

    return data

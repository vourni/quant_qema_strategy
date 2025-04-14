def optimize_strategy(self):
    # initializing ranges
    qema_range = range(15, 150, 5)
    fast_range = range(5, 60, 2)
    slow_range = range(60, 200, 5)

    # looping through all combinations of periods
    for period, fast, slow in product(qema_range, fast_range, slow_range):
        if fast >= slow:
            continue

        # getting data
        temp = self.data.copy()

        temp = self.apply_indicators(temp, period=period, fast_period=fast, slow_period=slow)
        temp = self.apply_signals(temp)
        temp = self.calculate_returns(temp)

        # calculating profit factor
        r = temp['strategy_return']
        pf = r[r>0].sum() / r[r<0].abs().sum()

        # checking profit factor
        if pf > self.pf:
            self.pf = pf
            self.qema_period, self.fast_period, self.slow_period = (period, fast, slow)
            self.best_data = temp

        # calculating sharpe ratio
        self.sharpe = (self.best_data['strategy_return'].mean() / self.best_data['strategy_return'].std()) * np.sqrt(48384)


    # calculating fast and slow emas
    data['fast_ema'] = data['hlc3'].ewm(span=fast_period, adjust=False).mean()
    data['slow_ema'] = data['hlc3'].ewm(span=slow_period, adjust=False).mean()



    # setting positon
    positions = [0]
    position = 0
    
    # looping through data
    for i in range(1, len(data)):
        # setting prev positon
        prev_pos = position
        
        # checking exit conditions
        if prev_pos == 1 and data['exit_long'].iloc[i] == 1:
            position = 0
        elif prev_pos == -1 and data['exit_short'].iloc[i] == 1:
            position = 0
        elif prev_pos == 0:
            position = data['raw_position'].iloc[i]
            
        positions.append(position)
        
    data['position'] = positions


        # setting exit signals
    data['exit_long'] = np.where((data['qema_distance_z'] > 2.5) |  
                               ((data['close'] < data['QEMA']) & (data['qema_slope'] < 0)), 1, 0)
    
    data['exit_short'] = np.where((data['qema_distance_z'] < -2.5) |  # 
                                ((data['close'] > data['QEMA']) & (data['qema_slope'] > 0)), 1, 0)
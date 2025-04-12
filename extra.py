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
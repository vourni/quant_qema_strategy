# Quantitative Strategy Benchmarking with Permutation Testing

This project implements a trading strategy on 1-hour equity data, using a custom-made breakout signal with trend-based filters and an ATR-based stop-loss. The strategy is validated against randomly permuted blocks of market data to assess whether performance is statistically significant or reacting to noise, then is tested in a classic test/train split to ensure there hasnt been issues of overfitting.

## Key Features

* Adaptive signal generation via custom indicator and its derivatives
* Trend filters for noise reduction
* ATR-based stop-loss with cooldown logic
* Hyperparameter tuning using Bayesian optimization
* Permutation testing to benchmark performance vs. randomized markets
* Train/Test split comparison testing to test for overfitting
* Visual output of Sharpe ratio, profit factor, hit rate distributions
* Calculation of performance metrics like win rate, average win, average loss, max drawdon, etc.

## Strategy Performance (QQQ, 2016–2025)

### Performance Against Permutations

| Metric             | Value    | Percentile vs. Permuted Data |
| ------------------ | -------- | ---------------------------- |
| Sharpe Ratio       | 1.79     | 100th percentile             |
| Profit Factor      | 1.22     | 99.5th percentile            |
| Hit Rate           | 53.15%   | 93.4th percentile            |

*See `strategy_plots_QQQ.pdf` for full results.*

### Performance Metrics

| Metric                                              | Value    |
| ----------------------------------------------------| -------- |
| Number of Trades                                    | 820      |
| Average Trade Length (hours)                        | 16.64    |
| Win Rate (After fees)                               | 52.67%   |
| Average Win (Percentage per hour of winning trade)  | 0.19%    |
| Average Loss (Percentage per hour of losing trade)  | -0.17%   |
| Arithmetic Return                                   | 9.81x    |
| Max Drawdown                                        | -9.48%   |
| Compounded Annual Growth Rate                       | 29.06%   |

*See `metrics_sc.png` for full results.*

### Train/Test Split Comparison

| Parameter          | Value    |
| ------------------ | -------- |
| Train Sharpe Ratio | 1.78     |
| Test Sharpe Ratio  | 1.35     |
| Train Proft Factor | 1.22     |
| Test Profit Factor | 1.2      |
| Train Hit Rate     | 52.95%   |
| Test Hit Rate      | 53.63%   |

*See `metrics_sc.png` for full results.*

### Best Parameters

| Parameter          | Value    |
| ------------------ | -------- |
| Q_Period           | 94       |
| D_Period           | 5        |
| Bars               | 1        |
| ATR Window         | 21       |
| ATR Multiplier     | 1.1      |
| Cooldown Period    | 3        |

*See `strat_metrics_sc.png` for full results.*

## Basis Performance

| Metric                                              | Value    |
| ----------------------------------------------------| -------- |
| Arithmetic Return                                   | 4.42x    |
| Max Drawdown                                        | -36.77%  |
| Compounded Annual Growth Rate                       | 19.85%   |

*See `basis_metrics_sc.png` for full results.*

## Project Structure

```
├── price_data              # Storage for price data
├── results                 # Storage for results
├── main.py                 # Main pipeline and testing framework
├── generate_signals.py     # Signal logic and filtering
├── data.csv                # Final strategy output (returns, positions)
```

## How to Run

1. Set up API credentials in a secret.py file (Alpaca keys)
2. Run:

```bash
python main.py
```

This will:

* Load or download price data
* Optimize parameters using Bayesian search
* Backtest strategy and evaluate performance
* Run permutation tests and generate plots

## Dependencies

* `pandas`
* `numpy`
* `matplotlib`
* `optuna`
* `alpaca_trade_api`

## License

MIT License

---

This project aims to provide a robust framework for testing the validity of trading strategies using rigorous statistical baselines.

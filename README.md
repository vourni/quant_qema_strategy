# Quantitative Strategy Benchmarking with Permutation Testing

This project implements a trading strategy on 1-hour equity data, using a custom-made breakout signal with trend-based filters and an ATR-based stop-loss. The strategy is validated against randomly permuted blocks of market data to assess whether performance is statistically significant or reacting to noise.

## Key Features

* Adaptive signal generation via custom indicator and its derivatives
* Trend filters for noise reduction
* ATR-based stop-loss with cooldown logic
* Hyperparameter tuning using Bayesian optimization
* Permutation testing to benchmark performance vs. randomized markets
* Visual output of Sharpe ratio, profit factor, hit rate distributions
* Calculation of performance metrics like win rate, average win, average loss, max drawdon, etc.

## Strategy Performance (TQQQ, 2016–2025)

### Performance Against Permutations

| Metric             | Value    | Percentile vs. Permuted Data |
| ------------------ | -------- | ---------------------------- |
| Sharpe Ratio       | 2.08     | 99.4th percentile            |
| Profit Factor      | 1.22     | 99.8th percentile            |
| Hit Rate           | 52.47%   | 99th percentile              |

*See `strategy_plots_TQQQ.pdf` for full results.*

### Performance Metrics

| Metric                                              | Value    |
| ----------------------------------------------------| -------- |
| Number of Trades                                    | 727      |
| Average Trade Length (hours)                        | 16.09    |
| Win Rate (After fees)                               | 51.83%   |
| Average Win (Percentage per hour of winning trade)  | 0.57%    |
| Average Loss (Percentage per hour of winning trade) | -0.52%   |
| Arithmetic Return                                   | 558.24x  |
| Max Drawdown                                        | -21.7%   |

*See `metrics_sc.png` for full results.*

### Best Parameters

| Parameter          | Value    |
| ------------------ | -------- |
| Q_Period           | 95       |
| D_Period           | 4        |
| Bars               | 3        |
| ATR Window         | 1.1      |
| ATR Multiplier     | 16       |
| Cooldown Period    | 3        |

*See `metrics_sc.png` for full results.*

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

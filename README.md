# Quantitative Strategy Benchmarking with Permutation Testing

This project implements a trading strategy on 1-hour equity data, using a custom-made breakout signal with volatility-based filters and ATR-based stop-loss. The strategy is validated against randomly permuted market data to assess whether performance is statistically significant.

## Key Features

* Adaptive signal generation via custom indicator and its derivatives
* Volatility and trend filters for noise reduction
* ATR-based stop-loss with cooldown logic
* Hyperparameter tuning using Bayesian optimization (Optuna)
* Permutation testing to benchmark performance vs. randomized markets
* Visual output of Sharpe ratio, profit factor, hit rate distributions

## Strategy Performance (TQQQ, 2016–2025)

| Metric             | Value    | Percentile vs. Permuted Data |
| ------------------ | -------- | ---------------------------- |
| Sharpe Ratio       | 1.91     | 90th percentile              |
| Profit Factor      | 1.20     | 99th percentile              |
| Hit Rate           | 52.48%   | 100th percentile             |
| Annualized Return  | 40.14%   | —                            |
| Avg Trade Duration | 17.3 hrs | —                            |

*See `strategy_plots_TQQQ.pdf` for full results.*

## Project Structure

```
.
├── main.py                  # Main pipeline and testing framework
├── generate_signals.py     # Signal logic and filtering
├── data.csv                # Final strategy output (returns, positions)
├── strategy_plots_TQQQ.pdf # Visual benchmark report
```

## How to Run

1. Set up API credentials (Alpaca keys)
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

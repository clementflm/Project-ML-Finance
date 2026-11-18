## Installation

Ensure Python 3.9+ is installed along with the required libraries:

```bash
pip install pandas numpy yfinance matplotlib scikit-learn
---

# Quantitative Trading Strategy: SPY Directional Prediction with Random Forest

## Project Overview
This project implements two end-to-end machine learning pipelines to predict next-day directional movements of the **S&P 500 ETF (SPY)**. The framework enforces chronological integrity to eliminate look-ahead bias and evaluates strategy viability net of transaction costs.
A Crossover Moving Average Strategy is also part of the tests, no ML for this one. It's less interestant but I still decided to keep it because it's what actually made me realized this project in ML.

---

## Performance Summary

![Strategy Performance Dashboard Time Series](Graphics/time_series_graph.png)

### Risk-Adjusted Metrics Comparison on the same period (2020 - 2026)
| Metric | Buy & Hold (SPY) | ML Strategy (Net of Fees) |
| :--- | :---: | :---: |
| **Annualized Return** | 17.60% | **17.05%** |
| **Annualized Volatility** | 17.11% | **16.61%** |
| **Sharpe Ratio** | 0.92 | **0.91** |
| **Maximum Drawdown** | -24.50% | **-22.63%** |

![Strategy Performance Dashboard Walk Forward](Graphics/walk_forward_graph.png)

### Risk-Adjusted Metrics Comparison (2010 - 2026)
| Metric | Buy & Hold (SPY) | ML Strategy (Net of Fees) |
| :--- | :---: | :---: |
| **Annualized Return** | 13.80% | **12.04%** |
| **Annualized Volatility** | 17.06% | **16.64%** |
| **Sharpe Ratio** | 0.73 | **0.65** |
| **Maximum Drawdown** | --33.72% | **-33.72%** |

![Strategy Performance Dashboard Moving Average](Graphics/strategy_mm_graph.png)

### Risk-Adjusted Metrics Comparison (1993 - 2026)
| Metric | Buy & Hold (SPY) | MM 29 / MM 199 Strategy (Net of Fees) |
| :--- | :---: | :---: |
| **Annualized Return** | 10.64% | **10.03%** |
| **Annualized Volatility** | 18.53% | **12.07%** |
| **Sharpe Ratio** | 0.53 | **0.65** |
| **Maximum Drawdown** | -55.19% | **-29.58%** |

*(Note: Strategies returns incorporate a 10 bps transaction cost penalty per trade)*

---

## Methodology & Rigor ML Strategies

1. **Features**
   * Raw non-stationary price series are transformed into stationary momentum, trend, and volatility indicators:
     * **Lagged Returns:** t, t-1, t-2, t-3 daily returns.
     * **Moving Average Ratios:** Short-to-medium and medium-to-long trend crossovers (`10/50`, `50/200`, `5/20`).
     * **Volatility:** Annualized 5-day and 30-day rolling standard deviations.
     * **Medium-Term Momentum** 20-day rate of change.

2. **Target Alignment & Look-Ahead Bias Prevention**
   * Targets are explicitly shifted (`shift(-1)`) so that features observed at market close t predict the return direction at t+1.
   * Trading signals are lagged by one period (`shift(1)`) during backtesting to ensure positions are only entered after signal generation.
   * Feature scaling (`StandardScaler`) is fit on training folds before transforming out-of-sample data.

3. **Hyperparameter Optimization & Validation**
   * Regularized `RandomForestClassifier` (`max_depth`, `min_samples_leaf`, `n_estimators`) via `GridSearchCV`.
   * Evaluated using **`TimeSeriesSplit`** and **`Walk-Forward`** validation.

---

## Repository Structure

```text
├── data/                   # Local CSV market data cache (ignored by Git)
├── Graphics/               # Exported high-resolution performance dashboards
│   └── dashboard_spy_rf.png
├── backtest_functions.py   # Risk metrics (Sharpe, Drawdown)
├── walk_forward.py         # ML training, Walk Forward
├── times_series.py         # ML training, Time Series strategy
├── strategy_mm.py          # Moving Average crossover strategy
├── requirements.txt        # Pinned Python dependencies for reproducibility
└── README.md










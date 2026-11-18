import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from backtest_functions import backtest_mm, max_drawdown, metrics
import os

file_path = "data/spy_raw.csv"
os.makedirs("data", exist_ok=True)

if os.path.exists(file_path):
    data = pd.read_csv("data/spy_raw.csv", header=[0, 1], index_col=0, parse_dates=True)
else:
    data = yf.download('SPY', start="1900-01-01", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.to_csv(file_path)

price = data["Close"].squeeze()

fig, ax = plt.subplots(figsize=(12, 6))

yields = price.pct_change().fillna(0)
cumul_market = (yields + 1).cumprod()
yields_market = cumul_market - 1
cumul_market.plot(ax=ax, label="Buy & Hold (SPY)", color="black", alpha=0.6)

test_upper = range(1, 50, 1)
test_lower = range(1, 200, 1)

results_array = []

# The condition low == up is to avoid the case where the two moving averages are equal, I still want when up < low or up > low, but not equal.
# The goal is also to see if the best strategy is buying the dip or buying the trend.
for up in test_upper:
    for low in test_lower:
        if low == up:
            continue
        result = backtest_mm(price, up, low)
        yields_strat = result[f"Strategy {up}/{low}"].iloc[-1] - 1
        results_array.append({'upper': up, 
                              'lower': low,
                              'yields_strat': yields_strat
                              })
        
df_optimisation = pd.DataFrame(results_array).sort_values(by='yields_strat', ascending=False)
best = df_optimisation.iloc[0]

backtest_best = backtest_mm(price, int(best['upper']), int(best['lower']))
yields_daily = backtest_best['Yields_Strategy_Gross'].fillna(0)

best_max_drawdown = max_drawdown(backtest_best[f"Strategy {int(best['upper'])}/{int(best['lower'])}"])

backtest_best[f"Strategy {int(best['upper'])}/{int(best['lower'])}"].plot(ax=ax, alpha = 0.8)

metrics_best = metrics(yields_daily)
metrics_market = metrics(yields)

df_metrics = pd.DataFrame({
    'Buy & Hold': metrics_market,
    f'Strategy MM{int(best['upper'])} / MM {int(best['lower'])}': metrics_best
})

df_display = df_metrics.copy()
df_display = df_display.astype(object)
df_display.loc['Yields Annualized'] = df_display.loc['Yields Annualized'].map('{:.2%}'.format)
df_display.loc['Volatility Annualized'] = df_display.loc['Volatility Annualized'].map('{:.2%}'.format)
df_display.loc['Sharpe Ratio'] = df_display.loc['Sharpe Ratio'].map('{:.2f}'.format)
df_display.loc['Max Drawdown'] = df_display.loc['Max Drawdown'].map('{:.2%}'.format)

print(df_display)

plt.title("Comparaison Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("$")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
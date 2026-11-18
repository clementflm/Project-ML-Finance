import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from backtest_functions import metrics
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
yields = price.pct_change()

X = pd.DataFrame({
    'return_t': yields,
    'return_t_1': yields.shift(1),
    'return_t_2': yields.shift(2),
    'return_t_3': yields.shift(3),
    'mm_ratio_10_50': price.rolling(10).mean() / price.rolling(50).mean(),
    'mm_ratio_50_200': price.rolling(50).mean() / price.rolling(200).mean(),
    'mm_ratio_5_20': price.rolling(5).mean() / price.rolling(20).mean(),
    'volatility_5d': yields.rolling(5).std() * (252 ** 0.5),
    'volatility_30d': yields.rolling(30).std() * (252 ** 0.5),
    'price_change': price.pct_change(20),
})

y = (yields.shift(-1) > 0).astype(int)

dataset = pd.concat([X, y.rename('target')], axis=1).dropna()

X_clean = dataset.drop(columns=['target'])
y_clean = dataset['target']

initial_train_size = int(len(X_clean) * 0.5)
step_size = 252

predictions = []
proba = []
indices = []

for start in range(initial_train_size, len(X_clean), step_size):
    end = min(start + step_size, len(X_clean))
    X_train = X_clean.iloc[:start]
    y_train = y_clean.iloc[:start]
    X_test = X_clean.iloc[start:end]
    y_test = y_clean.iloc[start:end]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    predictions.extend(y_pred)
    proba.extend(y_pred_proba)
    indices.extend(y_test.index)

y_pred_series = pd.Series(predictions, index=indices)
y_test_wf = y_clean.loc[indices]

print("=== Model Performance ===")
print(f"Accuracy (Test) : {accuracy_score(y_test_wf, y_pred_series):.2%}")
print("\nClassification Report :")
print(classification_report(y_test_wf, y_pred_series))

cm = confusion_matrix(y_test_wf, y_pred_series)

importance = pd.Series(model.feature_importances_, index=X_clean.columns).sort_values(ascending=True)

yields_test = yields.loc[y_test_wf.index].iloc[1:]  # Exclude the first day since we can't trade on it
signal_ml = y_pred_series.shift(1).iloc[1:]  # Shift the signal to avoid look-ahead bias and align with yields_test

trades = signal_ml.diff().abs().fillna(0)
print(f"Total trades executed: {int(trades.sum())}")

strat_ml_yields_gross = signal_ml * yields_test
strat_ml_yields_net = (1 + strat_ml_yields_gross) * (1 - trades * 0.001) - 1

cumul_strat_ml = (1 + strat_ml_yields_net).cumprod()
cumul_buy_hold = (1 + yields_test).cumprod()

cumul_gross = (1 + strat_ml_yields_gross).cumprod()
cost_trades = cumul_gross.iloc[-1] - cumul_strat_ml.iloc[-1]
ratio_cost = cost_trades / cumul_gross.iloc[-1]
print(f"The total cost of the trades is {cost_trades:.2f} which represents {ratio_cost:.2%} of the final price lost")

metrics_ml = metrics(strat_ml_yields_net)
metrics_buy_hold = metrics(yields_test)

df_metrics = pd.DataFrame({
    'Buy & Hold': metrics_buy_hold,
    'Strategy ML (Random Forest)': metrics_ml
})

df_display = df_metrics.copy()
df_display = df_display.astype(object)
df_display.loc['Yields Annualized'] = df_display.loc['Yields Annualized'].map('{:.2%}'.format)
df_display.loc['Volatility Annualized'] = df_display.loc['Volatility Annualized'].map('{:.2%}'.format)
df_display.loc['Sharpe Ratio'] = df_display.loc['Sharpe Ratio'].map('{:.2f}'.format)
df_display.loc['Max Drawdown'] = df_display.loc['Max Drawdown'].map('{:.2%}'.format)

print(df_display)

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.35, wspace=0.25)

ax0 = fig.add_subplot(gs[0, :])
cumul_buy_hold.plot(ax=ax0, label="Buy & Hold (SPY)", color="#717c7d", linewidth=1.5, linestyle="--")
cumul_strat_ml.plot(ax=ax0, label="Strategy ML (Random Forest) vs Buy & Hold (Net of Fees)", color="#2580c1", linewidth=2)

ax0.fill_between(cumul_strat_ml.index, cumul_strat_ml, cumul_buy_hold, 
                 where=(cumul_strat_ml >= cumul_buy_hold), color="#25c166", alpha=0.15, interpolate=True)
ax0.fill_between(cumul_strat_ml.index, cumul_strat_ml, cumul_buy_hold, 
                 where=(cumul_strat_ml < cumul_buy_hold), color="#e94735", alpha=0.15, interpolate=True)

ax0.set_title("Performance Out-of-Sample : Strategy ML (Random Forest) vs Buy & Hold (Net of Fees)", fontsize=13, fontweight="bold", pad=12)
ax0.set_ylabel("Capital", fontsize=11)
ax0.set_xlabel("")
ax0.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
ax0.grid(True, linestyle=":", alpha=0.6)

ax1 = fig.add_subplot(gs[1, 0])
importance.plot(ax=ax1, kind='barh', color="#253546", edgecolor='none', alpha=0.85)
ax1.set_title("Variable Importance", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Relative Importance", fontsize=10)
ax1.grid(True, axis='x', linestyle=":", alpha=0.6)
ax1.grid(False, axis='y')

ax2 = fig.add_subplot(gs[1, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Decrease (0)', 'Increase (1)'])
disp.plot(ax=ax2, cmap='Blues', colorbar=False, values_format='d')
ax2.set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Prediction Model", fontsize=10)
ax2.set_ylabel("Reality Market", fontsize=10)
ax2.grid(False)

fig.suptitle("Dashboard Performance : Strategy ML vs Buy & Hold", fontsize=15, fontweight="bold", y=0.98)

plt.show()
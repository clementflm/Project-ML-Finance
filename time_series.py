import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
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

# Train-test split (80% train, 20% test)

split_index = int(len(dataset) * 0.8)
X_train, X_test = X_clean.iloc[:split_index], X_clean.iloc[split_index:]
y_train, y_test = y_clean.iloc[:split_index], y_clean.iloc[split_index:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

time_series_split = TimeSeriesSplit(n_splits=4)

param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_leaf': [20, 30, 50, 60, 80, 100],
    'n_estimators': [100, 150, 200, 250, 300]
}

forests = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=forests,
    param_grid=param_grid,
    cv=time_series_split,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
model = grid_search.best_estimator_

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
y_pred_series = pd.Series(y_pred, index=y_test.index)

print("=== Model Performance ===")
print(f"Accuracy (Test) : {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report :")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

importance = pd.Series(model.feature_importances_, index=X_clean.columns).sort_values(ascending=True)

yields_test = yields.loc[y_test.index].iloc[1:]  # Exclude the first day since we can't trade on it
signal_ml = y_pred_series.shift(1).iloc[1:]  # Shift the signal to avoid look-ahead bias and align with yields_test

trades = signal_ml.diff().abs().fillna(0)

strat_ml_yields_gross = signal_ml * yields_test
strat_ml_yields_net = (1 + strat_ml_yields_gross) * (1 - trades * 0.001) - 1

cumul_strat_ml = (1 + strat_ml_yields_net).cumprod()
cumul_buy_hold = (1 + yields_test).cumprod()

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
cumul_buy_hold.plot(ax=ax0, label="Buy & Hold (SPY)", color="#7f8c8d", linewidth=1.5, linestyle="--")
cumul_strat_ml.plot(ax=ax0, label="Strategy ML (Random Forest) vs Buy & Hold (Net of Fees)", color="#1f77b4", linewidth=2)

ax0.fill_between(cumul_strat_ml.index, cumul_strat_ml, cumul_buy_hold, 
                 where=(cumul_strat_ml >= cumul_buy_hold), color="#2ecc71", alpha=0.15, interpolate=True)
ax0.fill_between(cumul_strat_ml.index, cumul_strat_ml, cumul_buy_hold, 
                 where=(cumul_strat_ml < cumul_buy_hold), color="#e74c3c", alpha=0.15, interpolate=True)

ax0.set_title("Performance Out-of-Sample : Strategy ML (Random Forest) vs Buy & Hold (Net of Fees)", fontsize=13, fontweight="bold", pad=12)
ax0.set_ylabel("Capital", fontsize=11)
ax0.set_xlabel("")
ax0.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
ax0.grid(True, linestyle=":", alpha=0.6)

ax1 = fig.add_subplot(gs[1, 0])
importance.plot(ax=ax1, kind='barh', color='#2c3e50', edgecolor='none', alpha=0.85)
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
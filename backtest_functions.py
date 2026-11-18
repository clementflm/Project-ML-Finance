import pandas as pd
import numpy as np

def backtest_mm(price, up, low, fee=0.001):
    """
    Does the backtest of a moving average crossover strategy.
    """
    yields_daily = price.pct_change().fillna(0)

    mm_up = price.rolling(up).mean()
    mm_low = price.rolling(low).mean()

    signal = (mm_up > mm_low).astype(int).shift(1).fillna(0)
    trades = signal.diff().abs().fillna(0)
    nb_trades = trades.cumsum()
    yields_strategy_gross = signal.mul(yields_daily, axis=0).fillna(0)
    yields_strategy_net = (1 + yields_strategy_gross) * (1 - trades * fee) - 1
    cumul_strat = (1 + yields_strategy_net).cumprod()

    name_strat = f"Strategy {up}/{low}"

    result = pd.DataFrame({
        'Yields_Strategy_Gross': yields_strategy_gross,
        'Yields_Strategy_Net': yields_strategy_net,
        name_strat: cumul_strat,
        'nb_trades': nb_trades
    })
    return result

def sharpe_ratio(yields_daily_net, rf=0.02):
    """
    Computes the Sharpe ratio of a series of yields.
    """
    rf_daily = (1 + rf)**(1/252) - 1
    yields_excess = yields_daily_net - rf_daily
    volatility_daily = yields_excess.std()
    if volatility_daily == 0 or pd.isna(volatility_daily):
        return 0
    ratio = (yields_excess.mean() / volatility_daily) * np.sqrt(252)
    
    return ratio

def max_drawdown(cumul):
    """
    Computes the maximum drawdown of a series of cumulative returns.
    """
    running_max = cumul.cummax()
    drawdown = (cumul / running_max) - 1
    max_drawdown = drawdown.min()
    
    return max_drawdown

def metrics(yields_daily_net, rf = 0.02):
    """
    Computes the Sharpe ratio, maximum drawdown, yields annualized, volatility of a series of daily yields.
    """
    sharpe = sharpe_ratio(yields_daily_net, rf)
    cumul = (1 + yields_daily_net).cumprod()
    max_dd = max_drawdown(cumul)
    volatility_year = yields_daily_net.std() * np.sqrt(252)
    n_days = len(yields_daily_net)
    yields_year = (cumul.iloc[-1]) ** (252 / n_days) - 1 if n_days > 0 else 0
    return {
        'Yields Annualized': yields_year,
        'Volatility Annualized': volatility_year,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }
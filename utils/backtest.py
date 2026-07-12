import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(actual_prices, predicted_prices, dates, initial_capital=100000.0, transaction_cost=0.001, threshold=0.0025):
    """
    Simulates a long-only trading strategy based on price predictions.
    
    Strategy:
    - Buy/Hold stock if tomorrow's predicted close is higher than today's actual close by > threshold.
    - Sell/Hold cash if tomorrow's predicted close is not higher by > threshold.
    
    Parameters:
    - actual_prices: 1D array of actual Close prices.
    - predicted_prices: 1D array of predicted Close prices.
    - dates: pandas Index/Series of dates.
    - initial_capital: starting capital in cash.
    - transaction_cost: fee percentage (e.g. 0.001 = 0.1% per trade).
    - threshold: minimum predicted change to trigger a buy signal.
    """
    n = len(actual_prices)
    cash = initial_capital
    shares = 0.0
    
    portfolio_values = []
    positions = [] # 1 if holding stock, 0 if cash
    trades = [] # list of (buy_price, sell_price, profit_pct)
    
    buy_price = 0.0
    
    # Run simulation
    for t in range(n):
        current_actual = actual_prices[t]
        
        # Determine signal for tomorrow (based on t's actual vs t+1's prediction)
        if t < n - 1:
            pred_tomorrow = predicted_prices[t+1]
            # Buy signal if predicted tomorrow is higher than today by threshold
            signal = 1 if pred_tomorrow > current_actual * (1 + threshold) else 0
        else:
            # Last day, force exit to close out portfolio
            signal = 0
            
        # Execute signals
        if signal == 1 and shares == 0:
            # Buy Stock
            shares = (cash * (1 - transaction_cost)) / current_actual
            cash = 0.0
            buy_price = current_actual
            positions.append(1)
        elif signal == 0 and shares > 0:
            # Sell Stock
            cash = (shares * current_actual) * (1 - transaction_cost)
            shares = 0.0
            profit_pct = ((current_actual - buy_price) / buy_price) - (2 * transaction_cost)
            trades.append(profit_pct)
            positions.append(0)
        else:
            # Hold current position
            positions.append(1 if shares > 0 else 0)
            
        # Record end-of-day portfolio value
        port_val = cash + (shares * current_actual)
        portfolio_values.append(port_val)
        
    portfolio_values = np.array(portfolio_values)
    
    # Benchmark strategy: Buy and Hold from day 0
    benchmark_shares = (initial_capital * (1 - transaction_cost)) / actual_prices[0]
    benchmark_values = actual_prices * benchmark_shares
    
    # Calculate performance metrics
    final_value = portfolio_values[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    bench_final_value = benchmark_values[-1]
    bench_return = ((bench_final_value - initial_capital) / initial_capital) * 100
    
    # Daily returns for Sharpe Ratio
    df_returns = pd.Series(portfolio_values).pct_change().dropna()
    bench_returns = pd.Series(benchmark_values).pct_change().dropna()
    
    # Annualized Sharpe Ratio (assuming 252 trading days and 5% risk free rate)
    daily_rf = 0.05 / 252
    excess_returns = df_returns - daily_rf
    if len(excess_returns) > 1 and df_returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() / df_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        
    if len(bench_returns) > 1 and bench_returns.std() > 0:
        bench_sharpe = ((bench_returns - daily_rf).mean() / bench_returns.std()) * np.sqrt(252)
    else:
        bench_sharpe = 0.0
        
    # Maximum Drawdown
    peaks = np.maximum.accumulate(portfolio_values)
    drawdowns = (peaks - portfolio_values) / peaks
    max_drawdown = drawdowns.max() * 100
    
    bench_peaks = np.maximum.accumulate(benchmark_values)
    bench_drawdowns = (bench_peaks - benchmark_values) / bench_peaks
    bench_max_drawdown = bench_drawdowns.max() * 100
    
    # Win Rate of trades
    num_trades = len(trades)
    winning_trades = sum(1 for t in trades if t > 0)
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0.0
    
    results = {
        'dates': dates,
        'portfolio_values': portfolio_values,
        'benchmark_values': benchmark_values,
        'final_value': final_value,
        'total_return': total_return,
        'bench_return': bench_return,
        'sharpe_ratio': sharpe_ratio,
        'bench_sharpe': bench_sharpe,
        'max_drawdown': max_drawdown,
        'bench_max_drawdown': bench_max_drawdown,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'trades': trades,
        'positions': positions
    }
    
    return results

def plot_backtest(results, save_path=None):
    """
    Plots the strategy equity curve vs Buy and Hold benchmark.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results['dates'], results['portfolio_values'], label=f"Model Strategy (Return: {results['total_return']:.2f}%)", color='#1f77b4', linewidth=2)
    plt.plot(results['dates'], results['benchmark_values'], label=f"Buy & Hold Benchmark (Return: {results['bench_return']:.2f}%)", color='#ff7f0e', linestyle='--', linewidth=1.5)
    
    plt.title("Trading Backtest: Strategy vs Benchmark Equity Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value (INR)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left')
    
    # Highlight performance metrics in text box
    stats_text = (
        f"--- Strategy Stats ---\n"
        f"Total Return: {results['total_return']:.2f}%\n"
        f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
        f"Trades Executed: {results['num_trades']}\n"
        f"Trade Win Rate: {results['win_rate']:.2f}%\n\n"
        f"--- Benchmark Stats ---\n"
        f"Buy & Hold Return: {results['bench_return']:.2f}%\n"
        f"Sharpe Ratio: {results['bench_sharpe']:.2f}\n"
        f"Max Drawdown: {results['bench_max_drawdown']:.2f}%"
    )
    plt.gca().text(
        0.02, 0.05, stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
        fontsize=9,
        verticalalignment='bottom'
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved backtest plot to {save_path}")
    plt.show()

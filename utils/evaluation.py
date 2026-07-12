import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(actual_original, pred_original, scaler, y_test, train_scaled, test_df=None, window_size=10, save_dir=None, model_name="Model"):

    # ---------- Convert to numpy ----------
    actual_vals = np.asarray(actual_original).reshape(-1)
    pred_vals = np.asarray(pred_original).reshape(-1)

    n = min(len(actual_vals), len(pred_vals))
    actual_vals = actual_vals[:n]
    pred_vals = pred_vals[:n]

    # ---------- Date handling ----------
    if test_df is not None:
        date_vals = pd.to_datetime(test_df['Date']).iloc[window_size:window_size+n].reset_index(drop=True)
    else:
        date_vals = pd.RangeIndex(start=0, stop=n)

    eval_df = pd.DataFrame({
        'Date': date_vals,
        'Actual': actual_vals,
        'Predicted': pred_vals
    })

    # ---------- Errors ----------
    eval_df['Error'] = eval_df['Actual'] - eval_df['Predicted']
    eval_df['AbsError'] = eval_df['Error'].abs()
    eval_df['APE_%'] = (eval_df['AbsError'] / np.clip(np.abs(eval_df['Actual']), 1e-8, None)) * 100

    # ---------- Metrics ----------
    rmse = np.sqrt(mean_squared_error(eval_df['Actual'], eval_df['Predicted']))
    mae = mean_absolute_error(eval_df['Actual'], eval_df['Predicted'])
    mape = eval_df['APE_%'].mean()
    r2 = r2_score(eval_df['Actual'], eval_df['Predicted'])

    # Direction accuracy
    actual_diff = np.diff(eval_df['Actual'])
    pred_diff = np.diff(eval_df['Predicted'])
    dir_acc = (np.sign(actual_diff) == np.sign(pred_diff)).mean() * 100

    print("\nFinal Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
  

    # ===================== PLOTS ===================== #
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Actual vs Predicted
    plt.figure(figsize=(14,6))
    plt.plot(eval_df['Date'], eval_df['Actual'], label='Actual')
    plt.plot(eval_df['Date'], eval_df['Predicted'], label='Predicted')
    plt.plot(eval_df['Date'], eval_df['Actual'].rolling(10).mean(), '--', label='Actual MA(10)')
    plt.plot(eval_df['Date'], eval_df['Predicted'].rolling(10).mean(), '--', label='Predicted MA(10)')
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{model_name}_actual_vs_predicted.png"), dpi=150)
    plt.show()

    # 2. Residual Plot
    plt.figure(figsize=(14,4))
    plt.plot(eval_df['Date'], eval_df['Error'])
    plt.axhline(0, linestyle='--')
    plt.title("Residuals Over Time")
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{model_name}_residuals.png"), dpi=150)
    plt.show()

    # 3. Scatter Plot
    plt.figure(figsize=(6,6))
    plt.scatter(eval_df['Actual'], eval_df['Predicted'], alpha=0.5)
    min_val = min(eval_df['Actual'].min(), eval_df['Predicted'].min())
    max_val = max(eval_df['Actual'].max(), eval_df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title("Actual vs Predicted Scatter")
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{model_name}_scatter.png"), dpi=150)
    plt.show()

    # 4. Error Distribution
    plt.figure(figsize=(6,4))
    plt.hist(eval_df['Error'], bins=30)
    plt.title("Error Distribution")
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{model_name}_error_dist.png"), dpi=150)
    plt.show()

    # 5. Cumulative Error
    plt.figure(figsize=(12,4))
    plt.plot(eval_df['Date'], eval_df['Error'].cumsum())
    plt.title("Cumulative Error")
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{model_name}_cumulative_error.png"), dpi=150)
    plt.show()

    # 6. Direction Plot
    plt.figure(figsize=(12,4))
    plt.plot(np.sign(actual_diff), label="Actual Direction")
    plt.plot(np.sign(pred_diff), label="Predicted Direction")
    plt.legend()
    plt.title("Direction Comparison")
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{model_name}_direction_comparison.png"), dpi=150)
    plt.show()

    return eval_df
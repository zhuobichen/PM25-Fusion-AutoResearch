# -*- coding: utf-8 -*-
"""
评估指标工具
============
提供统一的评估指标计算函数。

使用方式：
    from shared.metrics import compute_metrics

    metrics = compute_metrics(y_true, y_pred)
    print(metrics['R2'], metrics['RMSE'], metrics['MAE'], metrics['MB'])
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred):
    """
    计算 PM2.5 融合方法的评估指标。

    参数:
        y_true: 真实观测值 (array-like)
        y_pred: 预测值 (array-like)

    返回:
        dict: {'R2': float, 'MAE': float, 'RMSE': float, 'MB': float}
              全部为 float 类型，便于 JSON 序列化。
              如果无有效数据，返回 NaN。
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MB': np.nan}

    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MB': float(np.mean(y_pred - y_true))
    }

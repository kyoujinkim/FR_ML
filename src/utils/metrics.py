import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def dirMSE(pred, true):
    pred_diff = pred[:, 1:] - pred[:, :-1]
    true_diff = true[:, 1:] - true[:, :-1]
    return np.mean((pred_diff - true_diff) ** 2)

def CE(pred, true):
    pred = np.clip(pred, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)
    dirmse = dirMSE(pred, true)
    cross_entropy = CE(pred, true)

    return mae, mse, rmse, mape, mspe, dirmse, cross_entropy
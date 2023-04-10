import numpy as np
import scipy


def calc_errors(y_pred, y):
    errors = y_pred - y
    return errors
# Define statistics functions to use, with docstrings used as titles
def mean_signed_error(y_pred, y):
    "Mean signed error"
    errors = calc_errors(y_pred, y)
    return np.mean(errors)
def root_mean_squared_error(y_pred, y):
    "Root mean squared error"
    errors = calc_errors(y_pred, y)
    return np.sqrt((errors**2).mean())
def mean_unsigned_error(y_pred, y):
    "Mean unsigned error"
    errors = calc_errors(y_pred, y)
    return np.abs(errors).mean()
def kendall_tau(y_pred, y):
    "Kendall tau"
    return scipy.stats.kendalltau(y_pred, y)[0]
def pearson_r(y_pred, y):
    "Pearson R"
    return scipy.stats.pearsonr(y_pred, y)[0]   
# https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.mstats import pearsonr

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) / \
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


# def nrmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     """ Normalized Root Mean Squared Error """
#     if actual.max() - actual.min() < 1:
#         return None
#     return rmse(actual, predicted) / (actual.max() - actual.min())


def nrmse(actual: np.ndarray, predicted: np.ndarray, errors=None, nrmse_norm='quantile', **kwargs):
    if errors is None:
        sample_weights = np.ones(len(actual))
    else:
        sample_weights = 1. / errors ** 2

    rmse_tmp = mean_squared_error(actual, predicted, sample_weight=sample_weights)

    norm_vals = 0
    if nrmse_norm == 'std':
        norm_vals = np.std(actual)
    elif nrmse_norm == 'quantile':
        norm_vals = np.quantile(actual, 0.75) - np.quantile(actual, 0.25)
    elif nrmse_norm == 'range':
        actual.max() - actual.min()

    if norm_vals == 0:
        return None
    return 100 * np.sqrt(rmse_tmp) / norm_vals


def me(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Error """
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Geometric Mean Absolute Error """
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Median Absolute Error """
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Percentage Error """
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Absolute Percentage Error

    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Median Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def smdape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Median Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def maape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Arctangent Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1, **kwargs):
    """
    Mean Absolute Scaled Error

    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))


def std_ae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae)) / (len(actual) - 1))


def std_ape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape)) / (len(actual) - 1))


def rmspe(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Root Mean Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Root Median Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1, **kwargs):
    """ Root Mean Squared Scaled Error """
    q = np.abs(_error(actual, predicted)) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Integral Normalized Root Squared Error """
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))


def rrse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None, **kwargs):
    """ Mean Relative Error """
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None, **kwargs):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None, **kwargs):
    """ Median Relative Absolute Error """
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None, **kwargs):
    """ Geometric Mean Relative Absolute Error """
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None, **kwargs):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None, **kwargs):
    """ Unscaled Mean Bounded Relative Absolute Error """
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def redchisqg(actual: np.ndarray, predicted: np.ndarray, deg_free=2, sd=None, **kwargs):
    """
    Returns the reduced chi-square error statistic for an arbitrary model,
    chisq/nu, where nu is the number of degrees of freedom. If individual
    standard deviations (array sd) are supplied, then the chi-square error
    statistic is computed as the sum of squared errors divided by the standard
    deviations. See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    ydata,ymod,sd assumed to be Numpy arrays. deg integer.

    Usage:
    >>> chisq=redchisqg(actual,predicted,deg_free,sd)
    where
    ydata : data
    ymod : model evaluated at the same x points as ydata
    n : number of free parameters in the model
    sd : uncertainties in ydata

    Rodrigo Nemmen
    http://goo.gl/8S1Oo
     """
    # Chi-square statistic
    if sd is None:
        chisq = np.sum((actual - predicted) ** 2)
    else:
        chisq = np.sum(((actual - predicted) / sd) ** 2)

    # Number of degrees of freedom assuming 2 free parameters
    nu = actual.size - 1 - deg_free
    if nu < 1:
        return None
    # if np.isinf(res):
    #     res = None
    return chisq / nu


# https://stackoverflow.com/questions/42033720/python-sklearn-multiple-linear-regression-display-r-squared
# def r_squared(actual: np.ndarray, predicted: np.ndarray, **kwargs):
#     residual = sum((actual - predicted) ** 2)
#     total = sum((actual - np.mean(actual)) ** 2)
#     try:
#         res = 1 - (float(residual)) / total
#     except ZeroDivisionError:
#         return None
#     return res

def r_squared(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return r2_score(actual, predicted)


# https://stackoverflow.com/questions/42033720/python-sklearn-multiple-linear-regression-display-r-squared
def adj_rsquare(actual: np.ndarray, predicted: np.ndarray, deg_free, **kwargs):
    if len(actual) - deg_free - 1 < 1:
        return None
    r_square = r_squared(actual, predicted)
    res = 1 - (1 - r_square) * (len(actual) - 1) / (len(actual) - deg_free - 1)
    return res


METRICS = {
    'mse'         : mse,
    'rmse'        : rmse,
    'nrmse'       : nrmse,
    'me'          : me,
    'mae'         : mae,
    'mad'         : mad,
    'gmae'        : gmae,
    'mdae'        : mdae,
    'mpe'         : mpe,
    'mape'        : mape,
    'mdape'       : mdape,
    'smape'       : smape,
    'smdape'      : smdape,
    'maape'       : maape,
    'mase'        : mase,
    'std_ae'      : std_ae,
    'std_ape'     : std_ape,
    'rmspe'       : rmspe,
    'rmdspe'      : rmdspe,
    'rmsse'       : rmsse,
    'inrse'       : inrse,
    'rrse'        : rrse,
    'mre'         : mre,
    'rae'         : rae,
    'mrae'        : mrae,
    'mdrae'       : mdrae,
    'gmrae'       : gmrae,
    'mbrae'       : mbrae,
    'umbrae'      : umbrae,
    'mda'         : mda,
    'redchisqg'   : redchisqg,
    'r_squared'   : r_squared,
    'adj_rsquare' : adj_rsquare
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=None, **kwargs):
    if metrics is None or actual is None or predicted is None:
        return None
    results = {}
    for name in metrics:
        try:
            results[name] = globals()[name](actual, predicted, **kwargs)
            if results[name] is not None:
                if np.isinf(results[name]):
                    results[name] = None
        except Exception as err:
            results[name] = None
            print('Unable to compute metric {0}: {1}'.format(name, err))

    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))


def pearson_corr(xdata, ydata, **kwargs):
    nas = np.logical_or(xdata.isna(), ydata.isna())
    corr, _ = pearsonr(xdata[~nas], ydata[~nas])
    return corr

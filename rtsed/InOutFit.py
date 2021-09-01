import pandas as pd
import numpy as np
from scipy.stats.distributions import t
from scipy.optimize import curve_fit
from .common_fncts import merge_dicts
from datetime import datetime
import rtsed.forecasting_metrics as fm
from random import uniform
import rtsed.thermal_sed as ts


class InOutFit:
    _def_metrics = ('nrmse', 'smape', 'mape', 'mse', 'redchisqg', 'r_squared')
    _def_init_diam = 1.
    _def_bounds = (1.e-6, [1.e4, 30])

    def __init__(self, obj_id, fit_parameters):
        self._id = obj_id
        self._params = fit_parameters
        self._xdata = None
        self._ydata = None
        self._xdata_err = None
        self._ydata_err = None
        self._metadata = {}
        self._predicted = None
        self._bounds = None
        self._init_pars = None
        self.fit_results = None
        self.fit_errors = None
        self.fit_pcov = None
        self.metrics = None
        self._fitter = ts.flux_nu
        self._beams = None
        self._bounds = self._def_bounds

    def set_data_pandas(self, df: pd.DataFrame, xcol: str, ycol: str, xcol_err: str = None, ycol_err: str = None,
                        **kwargs):
        """
        Import data neaded for fitting from pandas dataframe
        @return:
        @param df: input dataframe, required
        @param xcol: string, name of the column for the xdata, required
        @param ycol: string, name of the column for the ydata, required
        @param xcol_err: string, name of the column for the ydata uncertanites, optional
        @param ycol_err: string, name of the column for the ydata uncertanites, optional
        @param kwargs: other columns needed for the fit, the values of dataframe[value] are stored in
                self._metadata[parameter]
        @return: self
        """
        self._xdata = df[xcol]
        self._ydata = df[ycol]
        if xcol_err is not None:
            self._xdata_err = df[xcol_err]
        if ycol_err is not None:
            self._ydata_err = df[ycol_err]
        for key, val in kwargs.items():
            self._metadata[key] = df[val]
        return self

    def set_bounds(self, bounds: tuple):
        """
        set bounds fot curve_fit fitter
        @param bounds: tuple, bounds for the fit
        @return: self
        """
        self._bounds = bounds
        return self

    @property
    def fit_results(self):
        """
        @return: dict, fited and fixed parameters {parameter:value}
        """
        return self._fit_results

    @fit_results.setter
    def fit_results(self, fit_results):
        self._fit_results = fit_results

    @property
    def fit_errors(self):
        """
        @return: dict, fit errors {parameter:value}
        """
        return self._fit_errors

    @fit_errors.setter
    def fit_errors(self, fit_errors):
        self._fit_errors = fit_errors

    @property
    def fit_pcov(self):
        return self._fit_pcov

    @fit_pcov.setter
    def fit_pcov(self, pcov):
        self._fit_pcov = pcov

    @property
    def full_results(self, return_metrics=_def_metrics):
        self.metrics = return_metrics
        res = merge_dicts(
            [{'id': self._id},
             self._params,
             self._init_pars,
             self.fit_results,
             self.fit_errors,
             self.metrics])
        res['n_points'] = len(self._ydata)
        res['time_in'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return res

    def set_predicted_values(self, fitter, xdata, **fit_results):
        if fit_results is None:
            self._predicted = None
        else:
            self._predicted = fitter(xdata, **fit_results)
        return self

    ###########################################################################
    # metrics and errors
    ###########################################################################

    def calc_fit_err(self, t_stat=False, conf_lev=0.2):
        sigma_ab = np.sqrt(np.diag(self.fit_pcov))
        dof = max(0, len(self._ydata) - len(self.fit_results))  # number of degrees of freedom
        tval = 1.
        if t_stat:
            tval = t.ppf(1.0 - conf_lev / 2.0, dof)  # student-t value for the dof and confidence level
        return sigma_ab * tval

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):

        parameters = {
            'actual'     : self._ydata,
            'predicted'  : self._predicted,
            'errors'     : self._ydata_err,
            'nrmse_norm ': 'quantile',
            'deg_free'   : 2
        }

        self._metrics = fm.evaluate(metrics=metrics, **parameters)

    ###########################################################################
    # fitting algorithm
    ###########################################################################

    def run_fit(self, **kwargs):

        self.set_init_pars()

        try:
            fit_results, self.fit_pcov = curve_fit(
                lambda x, theta, freq_0: self._fitter(x, theta,
                                                      freq_0,
                                                      self._params['mu'],
                                                      self._params['Te'],
                                                      self._params['model']),
                self._xdata,
                self._ydata,
                sigma=self._ydata_err,
                p0=list(self._init_pars.values()),
                bounds=self._bounds,
                **kwargs
            )

        except (ValueError, RuntimeError) as err:
            print("Fit didn't converge. Error:{}".format(err))
            return self

        self.fit_results = {"theta" : fit_results[0],
                            "freq_0": fit_results[1],
                            "mu"    : self._params['mu'],
                            'Te'    : self._params['Te'],
                            'model' : self._params['model']}

        fit_errors = self.calc_fit_err()
        self.fit_errors = {
            'theta_err' : fit_errors[0],
            'freq_0_err': fit_errors[1]
        }
        self.set_predicted_values(self._fitter, self._xdata, **self.fit_results)
        return self

    ###########################################################################
    # initial parameters
    ###########################################################################

    def set_init_pars(self):
        init_diam = self._def_init_diam
        if 'beams' in self._metadata:
            if np.sum(np.isnan(self._metadata['beams'])) < len(self._metadata['beams']):
                init_diam = np.nanmin(self._metadata['beams']) * uniform(2, 4)
        res = {
            'theta_start' : init_diam,
            'freq_0_start': 1.
        }
        self._init_pars = res

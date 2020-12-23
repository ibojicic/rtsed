import pandas as pd
import plotnine as gg
from rtsed.ggPlots import  ggPlots
from rtsed import thermal_sed


class PlotSed(ggPlots):

    def __init__(self):
        super().__init__()


    def plt_data_points(self, data, plot_err_bars=False):
        self.elements = gg.geom_point(data, gg.aes(x="freq", y="flux"),
                                      shape="$\\bigcirc$",
                                      size=3,
                                      color="#000080",
                                      inherit_aes=False)
        if plot_err_bars:
            self.elements = self.plt_error_bars(data)

        return self


    def plt_error_bars(self, data):
        # set error bar limits
        inp_data = data.copy()
        inp_data['ymin'] = inp_data['flux'] - inp_data['flux_err']
        inp_data.loc[inp_data['ymin'] < 0, 'ymin'] = 1.e-3
        inp_data['ymax'] = inp_data['flux'] + inp_data['flux_err']
        res = gg.geom_errorbar(inp_data, gg.aes(x="freq", ymin="ymin", ymax="ymax"),
                               colour="#000080",
                               na_rm=True,
                               inherit_aes=False,
                               width=0.05)
        return res

    def plt_thermal_sed(self, xdata, fitted_pars, lntp="-", clr='black', plt_conf=False, fitted_err=None):
        x = self.freq_points(xdata, no_points=50)
        y = thermal_sed.flux_nu(x, **fitted_pars)
        data = pd.DataFrame({"x": x, "y": y})
        self.elements = gg.geom_line(data, gg.aes(x=x, y=y),
                                     linetype=lntp,
                                     colour=clr,
                                     size=0.5,
                                     inherit_aes=False)
        if plt_conf:
            self.elements = self.plt_thermal_sed_confidence(xdata, fitted_pars, fitted_err, clr)

        return self


    def plt_thermal_sed_confidence(self, xdata, fitted_pars, fitted_err, clr):

        if fitted_err is None:
            return

        # x = self.freq_points(xdata)

        fitted_pars_max = fitted_pars.copy()

        fitted_pars_max['theta'] = fitted_pars_max['theta'] + fitted_err['theta_err'] * 1.645
        fitted_pars_max['freq_0'] = fitted_pars_max['freq_0'] + fitted_err['freq_0_err'] * 1.645
        y_max = thermal_sed.flux_nu(xdata, **fitted_pars_max)

        fitted_pars_min = fitted_pars.copy()
        fitted_pars_min['theta'] = fitted_pars_min['theta'] - fitted_err['theta_err'] * 1.645
        fitted_pars_min['freq_0'] = fitted_pars_min['freq_0'] - fitted_err['freq_0_err'] * 1.645
        y_min = thermal_sed.flux_nu(xdata, **fitted_pars_min)

        data = pd.DataFrame({"x": xdata, "y_min": y_min, "y_max": y_max})
        return gg.geom_ribbon(data, gg.aes(x=xdata, ymin=y_min, ymax=y_max), fill=clr, alpha=0.05)


    def plot_pwr_law(self, xdata, intersect, power, **kwargs):
        x = self.freq_points(xdata, no_points=2)
        y = 10 ** intersect * (x ** power)
        data = pd.DataFrame({"x": x, "y": y})
        self.elements = gg.geom_line(data, gg.aes(x=x, y=y),
                                     inherit_aes=False,
                                     **kwargs)

        return self

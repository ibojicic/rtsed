import numpy as np
import pandas as pd
import plotnine as gg

from .common_fncts import diff_keys


class ggPlots:
    dummy_data = pd.DataFrame([1])

    def __init__(self, plotype='loglog'):
        self.config = {
            'font'            : 'Courier New',
            'logtick_length'  : 0.008,
            'axis_text_size'  : 12,
            'axis_title_size' : 14,
            'legend_text_size': 8,
            'xlabel'          : 'Freq',
            'ylabel'          : 'Flux',
            'fill'            : 'None',
            'shape'           : 'o',
            'shape_size'      : 2,
            'shape_stroke'    : 0.2,
            'pallete'         : 'Set1',
            'legend_pos'      : 'right',
            'lintick_drct'    : 'in'

        }
        self.base_data = self.dummy_data
        self.xlimits = None
        self.ylimits = None
        self.baseplot = plotype
        self._elements = []

    @property
    def base_data(self):
        return self._base_data

    @base_data.setter
    def base_data(self, base_data):
        self._base_data = base_data

    @property
    def baseplot(self):
        return self._baseplot

    @property
    def plotype(self):
        return self._plotype

    @baseplot.setter
    def baseplot(self, plotype):
        self._plotype = plotype
        if plotype == 'loglog':
            self._baseplot = self.loglog_ggplot
        elif plotype == 'linlin':
            self._baseplot = self.linlin_ggplot
        elif plotype == 'loglin':
            self._baseplot = self.loglin_ggplot

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, element):
        module = element.__module__
        if module in ['plotnine.geoms.geom_errorbarh', 'plotnine.geoms.geom_errorbar']:
            self._elements.insert(0, element)
        else:
            self._elements.append(element)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def set_config_pars(self, **kwargs):
        for key, val in kwargs.items():
            self._config[key] = val

    @property
    def xlimits(self):
        return self._xlimits

    @xlimits.setter
    def xlimits(self, xlimits):
        if xlimits is None:
            self._xlimits = None
        elif isinstance(xlimits, tuple):
            if self.plotype in ['linlin', 'linlog']:
                self._xlimits = xlimits
            else:
                self._xlimits = np.log10(xlimits)
        else:
            self._xlimits = None

    @property
    def ylimits(self):
        return self._ylimits

    @ylimits.setter
    def ylimits(self, ylimits):
        if ylimits is None:
            self._ylimits = None
        elif isinstance(ylimits, tuple):
            if self.plotype in ['linlin', 'loglin']:
                self._ylimits = ylimits
            else:
                self._ylimits = np.log10(ylimits)
        else:
            self._ylimits = None

    def set_limits_on_data(self, xdata, ydata, xfactor=(0.5, 2.), yfactor=(0.5, 2.)):
        self.xlimits = (min(xdata) * xfactor[0], max(xdata) * xfactor[1])
        self.ylimits = (min(ydata) * yfactor[0], max(ydata) * yfactor[1])
        return self

    @staticmethod
    def freq_points(xdata, min_fac=0.5, max_fac=2.5, no_points=100):
        xmin = min(xdata) * min_fac
        xmax = max(xdata) * max_fac
        if xmin < 0:
            xmin = 0
        x = np.power(10, np.linspace(np.log10(xmin), np.log10(xmax), no_points))
        return x

    @staticmethod
    def save_plot(plot, path, heightim=16, widthim=20, units='cm', family='Courier', dpi=300):

        plot.save(filename=path,
                  height=heightim,
                  width=widthim,
                  units=units,
                  # device=device,
                  dpi=dpi,
                  family=family,
                  verbose=False)

    def loglog_ggplot(self):
        setup = gg.ggplot(self.base_data) + \
                gg.scale_x_log10() + \
                gg.scale_y_log10() + \
                gg.theme_bw(base_family=self.config['font']) + \
                gg.annotation_logticks(sides="trbl",
                                       lengths=(3 * self.config['logtick_length'],
                                                2 * self.config['logtick_length'],
                                                1 * self.config['logtick_length'])) + \
                gg.labs(x=self.config['xlabel'], y=self.config['ylabel']) + \
                gg.theme(
                    axis_ticks_length=-0.1,
                    panel_grid_minor=gg.element_blank(),
                    panel_grid_major=gg.element_blank(),
                    axis_text=gg.element_text(size=self.config['axis_text_size']),
                    axis_title=gg.element_text(size=self.config['axis_title_size']),
                    legend_position=self.config['legend_pos'],
                    legend_text=gg.element_text(size=self.config['legend_text_size']),
                    strip_background=gg.element_rect(colour="white", fill="white"),
                    panel_spacing=0,
                    legend_title_align='left',
                    legend_box_just='left',
                    legend_box_margin=0,
                    panel_border=gg.element_rect(colour="black", fill=None, size=1),
                    axis_line=gg.element_line(colour="black", size=1, linetype="solid")) + \
                gg.coord_cartesian(xlim=self.xlimits, ylim=self.ylimits)

        return setup

    def linlin_ggplot(self):
        setup = gg.ggplot(self.base_data) + \
                gg.theme_bw(base_family=self.config['font']) + \
                gg.labs(x=self.config['xlabel'], y=self.config['ylabel']) + \
                gg.theme(
                    axis_ticks_direction_x=self.config['lintick_drct'],
                    axis_ticks_direction_y=self.config['lintick_drct'],
                    panel_grid_minor=gg.element_blank(),
                    panel_grid_major=gg.element_blank(),
                    axis_text=gg.element_text(size=self.config['axis_text_size']),
                    axis_title=gg.element_text(size=self.config['axis_title_size']),
                    legend_position="right",
                    legend_text=gg.element_text(size=self.config['legend_text_size']),
                    strip_background=gg.element_rect(colour="white", fill="white"),
                    panel_spacing=0,
                    panel_border=gg.element_rect(colour="black", fill=None, size=1),
                    axis_line=gg.element_line(colour="black", size=1, linetype="solid")) + \
                gg.coord_cartesian(xlim=self.xlimits, ylim=self.ylimits)

        return setup

    def loglin_ggplot(self):
        setup = gg.ggplot(self.base_data) + \
                gg.scale_x_log10() + \
                gg.scale_y_continuous() + \
                gg.annotation_logticks(sides="tb",
                                       lengths=(3 * self.config['logtick_length'],
                                                2 * self.config['logtick_length'],
                                                1 * self.config['logtick_length'])) + \
                gg.theme_bw(base_family=self.config['font']) + \
                gg.labs(x=self.config['xlabel'], y=self.config['ylabel']) + \
                gg.theme(
                    axis_ticks_direction_y=self.config['lintick_drct'],
                    axis_ticks_major_x=gg.element_blank(),
                    axis_ticks_minor_x=gg.element_blank(),
                    panel_grid_minor=gg.element_blank(),
                    panel_grid_major=gg.element_blank(),
                    axis_text=gg.element_text(size=self.config['axis_text_size']),
                    axis_title=gg.element_text(size=self.config['axis_title_size']),
                    legend_position=self.config['legend_pos'],
                    legend_text=gg.element_text(size=self.config['legend_text_size']),
                    strip_background=gg.element_rect(colour="white", fill="white"),
                    panel_spacing=0,
                    panel_border=gg.element_rect(colour="black", fill=None, size=1),
                    axis_line=gg.element_line(colour="black", size=1, linetype="solid")) + \
                gg.coord_cartesian(xlim=self.xlimits, ylim=self.ylimits)

        return setup

    def plot_out(self):
        setup = self.baseplot()
        for element in self.elements:
            setup = setup + element
        return setup

    @staticmethod
    def parse_params(params, aes_pars: dict, ord_pars: dict):
        for key in params:
            if key[0:4] == 'aes_':
                aes_pars[key[4:]] = params[key]
            else:
                ord_pars[key] = params[key]

        ord_pars = diff_keys(ord_pars, aes_pars)

        return aes_pars, ord_pars

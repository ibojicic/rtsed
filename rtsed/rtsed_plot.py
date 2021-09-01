import click
import common_fncts as cf
from PlotSed import PlotSed
import numpy as np

models_config = {
    'sphshell': {
        'lntp'    : '-',
        'clr'     : '#990000',
        'plt_conf': True},
    'plshell' : {
        'lntp'    : 'dashdot',
        'clr'     : '#006600',
        'plt_conf': False}
}


@click.command()
@click.argument('input_file', nargs=1)
@click.argument('results_file', nargs=1)
@click.option('--out_format', '-f', default='png', help="Format of the output file (default is png).")
def cli(input_file, results_file, out_format):
    df_data = cf.read_input_file(input_file)

    df_results = cf.read_input_file(results_file)

    for curr_id in df_results.id.unique():
        print("Working on {}".format(curr_id))

        data_radio = df_data[df_data.id == curr_id]

        newnewplot = PlotSed()
        newnewplot.set_config_pars(xlabel='Freq (GHz)', ylabel='Flux (mJy)')
        newnewplot.set_limits_on_data(data_radio.freq, data_radio.flux)
        newnewplot.plt_data_points(data_radio, True)

        for mdl in df_results.model:
            fit_results = df_results.loc[
                (df_results.id == curr_id) & (df_results.model == mdl), ['theta', 'freq_0', 'mu', 'Te', 'model']]
            fit_errors = df_results.loc[
                (df_results.id == curr_id) & (df_results.model == mdl), ['theta_err', 'freq_0_err']]
            if np.isnan(fit_results.theta).all():
                continue
            newnewplot.plt_thermal_sed(data_radio.freq,
                                       fit_results.to_dict('r')[0],
                                       fitted_err=fit_errors.to_dict('r')[0],
                                       **models_config[mdl])

        newnewplot.save_plot(newnewplot.plot_out(),
                             "{}_sed.{}".format(
                                 curr_id, out_format))

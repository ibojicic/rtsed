import click
import common_fncts as cf
from InOutFit import InOutFit
import pandas as pd


@click.command()
@click.argument('input_file', nargs=1)
@click.argument('results_file', nargs=1)
@click.option('--models', '-m', default='all',
              help="Models for fitting, choose from sphshell, plshell or all (default is all).")
@click.option('--defunc', '-d', default=10, help="Flux unceirtainty if not measured (in %, default is 10%).")
@click.option('--te', '-t', default=1.E4, help="Electron temperature (in K, default is 1E4K).")
@click.option('--mu', '-u', default=0.4, help="Rin/Rout (default is 0.4).")
def cli(input_file, results_file, models, defunc, te, mu):
    if models == 'all':
        models = ['sphshell', 'plshell']
    else:
        models = [models]

    df_data = cf.read_input_file(input_file)
    df_data.flux_err.fillna(df_data.flux / defunc, inplace=True)

    full_result = pd.DataFrame()

    for curr_id in df_data.id.unique():
        print("Working on {}".format(curr_id))
        df_curr_data = df_data[df_data.id == curr_id].copy()

        for mdl in models:
            fit_params = {
                "model": mdl,
                "Te"   : te,
                "mu"   : mu
            }

            fitter = InOutFit(curr_id, fit_params)

            fitter.set_data_pandas(df_curr_data, xcol='freq', ycol='flux', ycol_err='flux_err', beams='bmaj')
            fitter.run_fit(absolute_sigma=True)
            result = fitter.full_results
            full_result = full_result.append(pd.DataFrame([result.values()], columns=result.keys()))


    full_result.to_csv(results_file, index=False)

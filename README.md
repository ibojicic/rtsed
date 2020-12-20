# rtsed

# Project Title

Radio-continuum spectral energy distribution (SED) fitter.

### Prerequisites

Rtsed fitter requires the usual packages (numpy, scipy, ...). Plotting is based on plotnine library. The packager will attempt to install all required packages, if it fails please install them manually.

For plotnine:

```
$ pip install plotnine         # 1. should be sufficient for most
$ pip install 'plotnine[all]'  # 2. includes extra/optional packages

# Or using conda
$ conda install -c conda-forge plotnine
```

### Installing

Installation can be done directly from github

```
pip install git+https://github.com/ibojicic/rtsed.git
```

You can test if the installation is succesful using the test input file in the tests/ folder. First download the file testfile.csv, cd to the folder where the file exists and try:

```
rtsed testfile.csv outfile.csv
```
This should produce file 'outfile.csv' in the current folder containing fit results (4 rows). 

To test plotting:

```
rtplot testfile.csv outfile.csv
```

should produce two png images ('389_sed.png' and '580_sed.png') in the current folder. If everything goes withour errors you're good to go :)

### Usage

To use the script you will need to provide .csv file containing radio data. The required columns are: *id, freq, band, flux, flux_err* and *bmaj* i.e. unique id of the object, frequency in GHz, integrated flux in mJy, uncertainty of the flux in mJy and beam size. The beam size parameter is used for initial diameter so, if you don't have it use a reasonable estimate. An example of the input file is provided in tests/ folder.

The basic usage for SED fitting is:

```
rtsed INPUT_FILE RESULTS_FILE
```

where IINPUT_FILE is (obviously) file containing your input radio data, and the RESULTS_FILE will be the name of the file containing resulta.

More advanced usage:

```
Usage: rtsed [OPTIONS] INPUT_FILE RESULTS_FILE

Options:
  -m, --models TEXT     Models for fitting, choose from sphshell, plshell or
                        all (default is all).

  -d, --defunc INTEGER  Flux unceirtainty if not measured (in %, default is
                        10%).

  -t, --te FLOAT        Electron temperature (in K, default is 1E4K).
  -u, --mu FLOAT        Rin/Rout (default is 0.4).
  --help                Show this message and exit.
```

For plotting of the fitted SED you can use rtplot script:

```
Usage: rtplot [OPTIONS] INPUT_FILE RESULTS_FILE

Options:
  -f, --out_format TEXT  Format of the output file (default is png).
  --help                 Show this message and exit.
```

## Author

**Ivan Bojicic** <qbocko@gmail.com> or <i.bojicic@westernsydney.edu.au>


## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details


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


## Author

* **Ivan Bojicic** 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc


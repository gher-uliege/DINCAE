[![documentation latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://gher-uliege.github.io/DINCAE/)
[![DOI](https://zenodo.org/badge/193079989.svg)](https://zenodo.org/badge/latestdoi/193079989)
[![codecov.io](http://codecov.io/github/gher-ulg/DINCAE/coverage.svg?branch=master)](http://codecov.io/github/gher-ulg/DINCAE?branch=master)

# DINCAE


DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations which is described in the following open access paper:
https://doi.org/10.5194/gmd-13-1609-2020


*Note that this code is no longer maintained and has been superseeded by https://github.com/gher-ulg/DINCAE.jl*



## Installation

Python 3.6 or 3.7 with the modules:
* numpy (https://docs.scipy.org/doc/numpy/user/install.html)
* netCDF4 (https://unidata.github.io/netcdf4-python/netCDF4/index.html)
* TensorFlow 1.15 with GPU support (https://www.tensorflow.org/install)

Tested versions:

* Python 3.6.8
* netcdf4 1.4.2
* numpy 1.15.4
* Tensorflow version 1.15 (DINCAE does not work with TensforFlow 2.0; TensorFlow 1.5 does not work on python 3.8)

You can install those packages either with `pip3` or with `conda`.


## Documentation

The document is available at https://gher-ulg.github.io/DINCAE/.

## Input format

The input data should be in netCDF with the variables:
* `lon`: longitude (degrees East)
* `lat`: latitude (degrees North)
* `time`: time (days since 1900-01-01 00:00:00)
* `mask`: boolean mask where true means the data location is valid
* `SST` (or any other varbiable name): the data


This is the example output from `ncdump -h`:

```
netcdf avhrr_sub_add_clouds {
dimensions:
	time = UNLIMITED ; // (5266 currently)
	lat = 112 ;
	lon = 112 ;
variables:
	double lon(lon) ;
	double lat(lat) ;
	double time(time) ;
		time:units = "days since 1900-01-01 00:00:00" ;
	int mask(lat, lon) ;
	float SST(time, lat, lon) ;
		SST:_FillValue = -9999.f ;
}
```

An example for how to create this file in the examples folder:
* [python example](https://github.com/gher-ulg/DINCAE/blob/master/examples/create\_input\_file.py)
* [julia example](https://github.com/gher-ulg/DINCAE/blob/master/examples/create\_input\_file.jl)


## Running DINCAE

Copy the template file `run_DINCAE.py` and adapt the filename, variable name and the output directory and possibly optional arguments for the reconstruction method as mentioned in the [documentation](https://gher-ulg.github.io/DINCAE/).
The code can be run as follows:

```bash
python3 run_DINCAE.py
```

The output NetCDF files are contain the variables:
* `meandata` the time average of the input data used to compute the anomalies
* `mean_rec` and `sigma_rec`: the mean and standard deviation of the Gaussian probability distribution function of the reconstruction.  

In [Barth et al., 2020](https://doi.org/10.5194/gmd-13-1609-2020) the best results were obtained by averaging all the NetCDF files.

## Reducing GPU memory

Convolutional neural networks can require "a lot" of GPU memory. These parameters can affect GPU memory utilisation:

* reduce the mini-batch size
* use fewer layers (e.g. `enc_nfilter_internal` = [16,24,36] or [16,24])
* use less filters (reduce the values of the optional parameter enc_nfilter_internal)
* reduce `frac_dense_layer`, a parameter controlling the width of the dense layer in the bottleneck
* use a smaller domain or lower resolution


## Example results

[Link to animation](http://data-assimilation.net/upload/Alex/DINCAE/data-avg-DINCAE-AVHRR.gif)


More information about this result is given in the [linked paper](https://www.geosci-model-dev-discuss.net/gmd-2019-128/).

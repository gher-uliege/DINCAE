[![documentation latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://gher-ulg.github.io/DINCAE/)

# DINCAE


DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations.


## Installation

Python 3.6 with the modules:
* numpy (https://docs.scipy.org/doc/numpy/user/install.html)
* netCDF4 (https://unidata.github.io/netcdf4-python/netCDF4/index.html)
* TensorFlow 1.12 with GPU support (https://www.tensorflow.org/install)

Tested versions:

* Python 3.6.8
* netcdf4 1.4.2
* numpy 1.15.4
* Tensorflow version 1.12.0

You can install those packages either with `pip3` or with `conda`.

## Input format

The input data should be in netCDF with the variables:
* `lon`: longitude (degrees East)
* `lat`: latitude (degrees North)
* `time`: time (days since 1900-01-01 00:00:00)
* `mask`: boolean mask where true means the data location is valid
* `SST` (or any other varbiable name): data


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

## Running DINCAE

Copy the template file `run_DINCAE.py` and adapt the filename, variable name and the output directory and possibly optional arguments for the reconstruction method as mentioned in the [documentation](https://gher-ulg.github.io/DINCAE/).
The code can be run as follows:

```bash
export PYTHONPATH=/path/to/module
python3 run_DINCAE.py
```

`/path/to/module` should be replaced by the directory name containing the file `DINCAE.py`.

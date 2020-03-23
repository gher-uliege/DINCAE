from netCDF4 import Dataset, num2date
import numpy as np

# change the variables below with your own data

# the filename
filename = "input_file_python.nc"
# the variable name in the netCDF file
varname = "SST"
# longitude in degrees East
lon = np.array([1,1.5,2])
# latitude in degrees West
lat = np.array([-1,0,1])
# time in days since a starting day
time = np.array([0,1,2])
# time units
time_units = "days since 1900-01-01 00:00:00";
# the data to reconstruct
data = np.zeros((len(time),len(lat),len(lon)))
# special value to indice missing data (cloud or land)
fill_value = -9999.
# mask, one means the data location is valid, e.g. a sea points for sea surface temperature.
mask = np.ones((len(lat),len(lon)))


# create the NetCDF file

root_grp = Dataset(filename, 'w', format='NETCDF4')

# dimensions
root_grp.createDimension('lon', len(lon))
root_grp.createDimension('lat', len(lat))
root_grp.createDimension('time', None)

# variables
nc_lon = root_grp.createVariable('lon', 'f4', ('lon',))
nc_lat = root_grp.createVariable('lat', 'f4', ('lat',))
nc_time = root_grp.createVariable('time', 'f4', ('time',))
nc_time.units = time_units

nc_data = root_grp.createVariable(varname, 'f4', ('time', 'lat', 'lon'),
                                  fill_value=fill_value)

nc_mask = root_grp.createVariable("mask", 'i4', ('lat', 'lon'))
nc_mask.comment = "one means the data location is valid (e.g. sea for SST), zero the location is invalid (e.g. land for SST)"

# data
nc_lon[:] = lon
nc_lat[:] = lat
nc_time[:] = time
nc_data[:,:] = data
nc_mask[:,:] = mask

root_grp.close()

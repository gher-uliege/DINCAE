using NCDatasets, DataStructures, Dates

# change the variables below with your own data

# the filename
filename = "input_file_julia.nc"
# the variable name in the netCDF file
varname = "SST"
# longitude in degrees East
lon = [1,1.5,2]
# latitude in degrees West
lat = [-1,0,1]
# time
time = [DateTime(1900,1,1),DateTime(1900,1,2),DateTime(1900,1,3)]
# time units
time_units = "days since 1900-01-01 00:00:00";
# the data to reconstruct
data = zeros((length(time),length(lat),length(lon)))
# special value to indice missing data (cloud or land)
fill_value = -9999.
# mask, one means the data location is valid, e.g. a sea points for sea surface temperature.
mask = ones((length(lat),length(lon)))


# create the NetCDF file


ds = NCDataset(filename,"c")

# Dimensions

ds.dim["lon"] = 3
ds.dim["lat"] = 3
ds.dim["time"] = Inf # unlimited dimension

# Declare variables

nclon = defVar(ds,"lon", Float32, ("lon",))

nclat = defVar(ds,"lat", Float32, ("lat",))

nctime = defVar(ds,"time", Float32, ("time",), attrib = OrderedDict(
    "units"                     => "days since 1900-01-01 00:00:00",
))

ncSST = defVar(ds,varname, Float32, ("lon", "lat", "time"), attrib = OrderedDict(
    "_FillValue"                => Float32(-9999.0),
))

ncmask = defVar(ds,"mask", Int32, ("lon", "lat"), attrib = OrderedDict(
    "comment"                   => "one means the data location is valid (e.g. sea for SST), zero the location is invalid (e.g. land for SST)",
))


# Define variables

nclon[:] = lon
nclat[:] = lat
nctime[:] = time
ncSST[:,:,:] = data
ncmask[:,:] = mask

close(ds)

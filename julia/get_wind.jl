using NCDatasets, PyPlot, OceanPlot, Dates, Interpolations, Statistics


lonr = [12,19]; latr = [40,46];
timer = [DateTime(2000,3,1),DateTime(2006,7,31)]

#timer = [DateTime(2000,3,1),DateTime(2000,3,2)]

times = timer[1]:Dates.Day(1):timer[end]

outfile = "/tmp/wnd.nc"
outfile = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/CCMP_Wind_Analysis.nc"

rr(ind) = ind[1]:ind[end]

function target_grid(lonr,latr)

    url = "https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/modis/L3/terra/11um/v2014.0/4km/daily/2000/055/T2000055.L3m_DAY_SST_sst_4km.nc"
    ds = Dataset(url)
    lon = nomissing(ds["lon"][:]);
    lat = nomissing(ds["lat"][:]);
    irange = rr(findall(lonr[1] .<= lon .<= lonr[end]));
    jrange = rr(findall(latr[1] .<= lat .<= latr[end]));

    lon_target = lon[irange];
    lat_target = lat[jrange];
    return lon_target,lat_target
end

lon_target,lat_target = target_grid(lonr,latr)

if isfile(outfile)
    rm(outfile)
end
dsout = Dataset(outfile,"c")
# Dimensions

dsout.dim["time"] = Inf # unlimited dimension
dsout.dim["lat"] = length(lat_target)
dsout.dim["lon"] = length(lon_target)

# Declare variables

nclon = defVar(dsout,"lon", Float64, ("lon",))
nclat = defVar(dsout,"lat", Float64, ("lat",))
nctime = defVar(dsout,"time", Float64, ("time",))
nctime.attrib["units"] = "days since 1900-01-01 00:00:00"
#ncmask = defVar(dsout,"mask", Int32, ("lon", "lat"))
ncuwnd = defVar(dsout,"uwnd", Float32, ("lon", "lat", "time"))
ncuwnd.attrib["_FillValue"] = Float32(-9999.0)

ncvwnd = defVar(dsout,"vwnd", Float32, ("lon", "lat", "time"))
ncvwnd.attrib["_FillValue"] = Float32(-9999.0)

# Define variables

nclon[:] = lon_target
nclat[:] = lat_target

for n = 1:length(times)
    dt = times[n]
    @show dt
    yyyy = Dates.format(dt,"yyyy")
    mm = Dates.format(dt,"mm")
    dd = Dates.format(dt,"dd")

    url = "http://data.remss.com/ccmp/v02.0/Y$(yyyy)/M$(mm)/CCMP_Wind_Analysis_$(yyyy)$(mm)$(dd)_V02.0_L3.0_RSS.nc"
    fname = download(url)
    #fname = "/tmp/juliaeiOWI2"

    #fname = "http://data.remss.com/ccmp/v02.0/Y1993/M07/CCMP_Wind_Analysis_19930701_V02.0_L3.0_RSS.nc"
    #fname = "/tmp/CCMP_Wind_Analysis_19930701_V02.0_L3.0_RSS.nc"

    ds = Dataset(fname)
    uwnd = nomissing(ds["uwnd"][:],NaN);
    vwnd = nomissing(ds["vwnd"][:],NaN);
    lon = nomissing(ds["longitude"][:]);
    lat = nomissing(ds["latitude"][:]);

    uwnd = mean(uwnd,dims = 3)[:,:,1]
    vwnd = mean(vwnd,dims = 3)[:,:,1]


    itpu = interpolate((lon,lat),uwnd,Gridded(Linear()));
    uwnd_target = itpu.(lon_target,lat_target');

    itpv = interpolate((lon,lat),vwnd,Gridded(Linear()));
    vwnd_target = itpv.(lon_target,lat_target');

    close(ds)
    rm(fname)

    nctime[n] = dt
    #ncmask[:] = ...
    ncuwnd[:,:,n] = uwnd_target
    ncvwnd[:,:,n] = vwnd_target

    sync(dsout)
end

close(dsout)


#=
s = sqrt.(uwnd.^2 + vwnd.^2);
clf();  quiver(lon[irange],lat[jrange],uwnd[irange,jrange,1]',vwnd[irange,jrange,1]',s[irange,jrange,1]')
colorbar()
using OceanPlot
OceanPlot.plotmap
OceanPlot.plotmap()
quiver(lon[irange],lat[jrange],uwnd[irange,jrange,1]',vwnd[irange,jrange,1]',s[irange,jrange,1]')
=#

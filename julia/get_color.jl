using NCDatasets, PyPlot, OceanPlot, Dates, Interpolations, Statistics, Printf


include("common.jl")

outfile = joinpath(basedir,"color.nc")

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

ncdata = defVar(dsout,"chlor_a", Float32, ("lon", "lat", "time"),
                deflatelevel = 9,
                chunksizes = [length(lon_target),length(lat_target),1])

ncdata.attrib["_FillValue"] = Float32(-9999.0)

# Define variables

nclon[:] = lon_target
nclat[:] = lat_target

for n = 1:length(times)
    dt = times[n]
    nctime[n] = dt

    url = data_url(dt)
    @show dt,url

    fname = ""
    for i = 1:3
        try
            fname = download(url)
            break
        catch
            @info "try again $url"
        end
    end
    #fname = "/tmp/juliaeiOWI2"

    #fname = "http://data.remss.com/ccmp/v02.0/Y1993/M07/CCMP_Wind_Analysis_19930701_V02.0_L3.0_RSS.nc"
    #fname = "/tmp/CCMP_Wind_Analysis_19930701_V02.0_L3.0_RSS.nc"

    if isfile(fname)
        ds = Dataset(fname)
        data = nomissing(ds["chlor_a"][:],NaN);
        lon = nomissing(ds["lon"][:]);
        lat = nomissing(ds["lat"][:]);

        if lat[1] > lat[2]
            lat = reverse(lat)
            data = reverse(data,dims=2)
        end

        # depending on domain
        #lon = lon .- 360

        itp = interpolate((lon,lat),data,Gridded(Linear()));
        data_target = itp.(lon_target,lat_target');

        close(ds)
        rm(fname)

        ncdata[:,:,n] = replace(data_target,NaN => missing)
    else
        @warn "no data $dt"
        ncdata[:,:,n] .= missing
    end
    sync(dsout)
end

close(dsout)

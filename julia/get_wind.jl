using NCDatasets, PyPlot, OceanPlot, Dates, Interpolations, Statistics, Printf

include("common.jl")

#outfile = joinpath(basedir,"CCMP_Wind_Analysis.nc")
outdir = joinpath(basedir,"CCMP_Wind_Analysis")
mkpath(outdir)

lon_target,lat_target = target_grid(lonr,latr)


for n = 1:length(times)
    dt = times[n]

    outfile = joinpath(outdir,"wind-$(dt).nc")
    if isfile(outfile)
        @info "already $outfile"
        continue
    end

    url = wind_url(dt)
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

    if !isfile(fname)
        @info "failled to download $url"
        continue
    end

    ds = Dataset(fname)
    uwnd = nomissing(ds["uwnd"][:],NaN);
    vwnd = nomissing(ds["vwnd"][:],NaN);
    lon = nomissing(ds["longitude"][:]);
    lat = nomissing(ds["latitude"][:]);

    uwnd = mean(uwnd,dims = 3)[:,:,1]
    vwnd = mean(vwnd,dims = 3)[:,:,1]

    # depending on domain
    if all(lonr .> 0)
        # ok
    elseif all(lonr .< 0)
        lon = lon .- 360
    else
        error("Huston, we have a problem")
    end

    itpu = interpolate((lon,lat),uwnd,Gridded(Linear()));
    uwnd_target = itpu.(lon_target,lat_target');

    itpv = interpolate((lon,lat),vwnd,Gridded(Linear()));
    vwnd_target = itpv.(lon_target,lat_target');

    close(ds)
    rm(fname)


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
    nctime[1] = dt
    #ncmask[:] = ...
    ncuwnd[:,:,1] = uwnd_target
    ncvwnd[:,:,1] = vwnd_target
    close(dsout)
end




#=
s = sqrt.(uwnd.^2 + vwnd.^2);
clf();  quiver(lon[irange],lat[jrange],uwnd[irange,jrange,1]',vwnd[irange,jrange,1]',s[irange,jrange,1]')
colorbar()
using OceanPlot
OceanPlot.plotmap
OceanPlot.plotmap()
quiver(lon[irange],lat[jrange],uwnd[irange,jrange,1]',vwnd[irange,jrange,1]',s[irange,jrange,1]')
=#

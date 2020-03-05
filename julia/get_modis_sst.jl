using NCDatasets, PyPlot, OceanPlot, Dates, Interpolations, Statistics, Printf


include("common.jl")

mkpath(joinpath(basedir,"modis"))

data_url = modis_url

#lon_target,lat_target = target_grid(lonr,latr)



url = data_url(times[1])
ds = Dataset(url)
lon = nomissing(ds["lon"][:]);
lat = nomissing(ds["lat"][:]);
irange = rr(findall(lonr[1] .<= lon .<= lonr[end]));
jrange = rr(findall(latr[1] .<= lat .<= latr[end]));
close(ds)


for n = 1:length(times)
#for n = 1:2
#for n = 950 .+ (0:1)
    dt = times[n]

    url = data_url(dt)
    @show dt,url

    outfile = joinpath(basedir,"modis","modis_sst_$(Dates.format(dt,"yyyy-mm-dd")).nc")

    if !isfile(outfile)
        # fname = ""
        # for i = 1:3
        #     try
        #         fname = download(url)
        #         break
        #     catch
        #         @info "try again $url"
        #     end
        # end
        #fname = "/tmp/juliaeiOWI2"

        #fname = "http://data.remss.com/ccmp/v02.0/Y1993/M07/CCMP_Wind_Analysis_19930701_V02.0_L3.0_RSS.nc"
        #fname = "/tmp/CCMP_Wind_Analysis_19930701_V02.0_L3.0_RSS.nc"

        if isfile(url)
            ds = Dataset(url)
            data = ds["sst"][irange,jrange,1];
            qual = ds["qual_sst"][irange,jrange,1]

            # keep only quality 0 (best) to 3
            #data[(qual .> 3) .| ismissing.(qual)] .= NaN
            # depending on domain
            #lon = lon .- 360

            if lat[1] > lat[2]
                data = reverse(data,dims=2)
                qual = reverse(qual,dims=2)
            end

            close(ds)
        else
            data = fill(-9999.0,(length(irange),length(jrange)))
            qual = fill(-9999.0,(length(irange),length(jrange)))
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

        ncdata = defVar(dsout,"sst", Float32, ("lon", "lat", "time"),
                        deflatelevel = 9,
                        chunksizes = [length(lon_target),length(lat_target),1])
        ncdata.attrib["_FillValue"] = Float32(-9999.0)

        ncqual = defVar(dsout,"qual", Float32, ("lon", "lat", "time"),
                        deflatelevel = 9,
                        chunksizes = [length(lon_target),length(lat_target),1])
        ncqual.attrib["_FillValue"] = Float32(-9999.0)

        # Define variables

        nclon[:] = lon_target
        nclat[:] = lat_target
        ncdata[:,:,1] = replace(data,NaN => missing)
        ncqual[:,:,1] = replace(qual,NaN => missing)
        nctime[1] = dt
        #else
        #    @warn "no data $dt"
        #    ncdata[:,:,n] .= missing
        #end
        close(dsout)
    end
end

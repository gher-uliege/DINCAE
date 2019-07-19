lonr = [12,19]
latr = [40,46];
timer = [DateTime(2000,3,1),DateTime(2006,7,31)] # wind
timer = [DateTime(2003,1,1),DateTime(2016,12,31)]
#timer = [DateTime(2003,1,1),DateTime(2003,1,2)] # test

times = timer[1]:Dates.Day(1):timer[end]

#lonr = [-16.5,-8]
#latr = [35,44];
#timer = [DateTime(2001,1,1),DateTime(2018,12,31)]
basedir = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/Adriatic2"
mkpath(basedir)

rr(ind) = ind[1]:ind[end]

function target_grid(lonr,latr)

    url = "https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/modis/L3/terra/11um/v2014.0/4km/daily/2000/055/T2000055.L3m_DAY_SST_sst_4km.nc"
    ds = Dataset(url)
    lon = nomissing(ds["lon"][:]);
    lat = nomissing(ds["lat"][:]);
    irange = rr(findall(lonr[1] .<= lon .<= lonr[end]));
    jrange = rr(findall(latr[1] .<= lat .<= latr[end]));
    close(ds)

    lon_target = lon[irange];
    lat_target = lat[jrange];
    return lon_target,lat_target
end

function color_url(dt)
    yyyy = Dates.format(dt,"yyyy")
    #mm = Dates.day(dt,"mm")
    #dd = Dates.format(dt,"dd")
    doy = @sprintf("%03d",Dates.dayofyear(dt))

    #url = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A2003001.L3m_DAY_CHL_chlor_a_4km.nc"
    url = "https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/A$(yyyy)$(doy).L3m_DAY_CHL_chlor_a_4km.nc"

    return url
end


function wind_url(dt)
    yyyy = Dates.format(dt,"yyyy")
    mm = Dates.format(dt,"mm")
    dd = Dates.format(dt,"dd")

    url = "http://data.remss.com/ccmp/v02.0/Y$(yyyy)/M$(mm)/CCMP_Wind_Analysis_$(yyyy)$(mm)$(dd)_V02.0_L3.0_RSS.nc"
    return url
end

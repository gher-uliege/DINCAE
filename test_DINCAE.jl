using DINCAE

fname = "avhrr_sub_add_clouds.nc"
varname = "SST"

lon,lat,datatime,data_full,missingmask,mask = DINCAE.load_gridded_nc(fname,varname)


#datagen,ntimes,meandata = DINCAE.data_generator(lon,lat,datatime,data_full,missingmask, train = false)
#xin,xtrue = datagen(i);


dd = DINCAE.NCData(lon,lat,datatime,data_full,missingmask)

i = 1;
xin,xtrue = dd[i];

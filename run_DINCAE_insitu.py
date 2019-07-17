#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import math
from math import ceil
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import DINCAE
import scipy
import skopt
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import json
import shutil

from multiprocessing import Pool

epochs = 5000*2
epochs = 1000
#epochs = 5

reconstruct_params = {
    #"epochs": 1,
    #"epochs": 1_000 * 5 * 2,
    "epochs": epochs,
    #"epochs": 5,
    "batch_size": 12,
    "skipconnections": [],
    #"save_each": 100 * 2,
    #"save_each": 5,
    "save_each": 0,
    "dropout_rate_train": 0.3,
    "shuffle_buffer_size": 12,
    "resize_method": tf.image.ResizeMethod.BILINEAR,
    "enc_ksize_internal": [16,24,36,54],
    #"regularization_L2_beta": 0.,
    "regularization_L2_beta": 0.05,
    "save_model_each": 0,
}


# basedir should contain all the input data
basedir = os.path.expanduser("~/Data/DINCAE_insitu/")

fnametrain = os.path.join(basedir,"Temperature.train.nc")
varname = "Salinity"
varname = "Temperature"

outdir = os.path.join(basedir,"Optimization5")


maskname = os.path.join(basedir,"mask.nc")

ds = Dataset(maskname, 'r')
lon = ds.variables["lon"][:].data;
lat = ds.variables["lat"][:].data;
mask = np.array(ds.variables["mask"][:,:].data,dtype = np.bool);

imax = len(lon)
jmax = len(lat)


if not os.path.isdir(basedir):
    os.mkdir(basedir)

if not os.path.isdir(outdir):
    os.mkdir(outdir)

def loadobs(fname,varname):

    ds = Dataset(fname)
    obslon = ds.variables["obslon"][:];
    obslat = ds.variables["obslat"][:];
    obsdepth = ds.variables["obsdepth"][:];
    obstime = ds.variables["obstime"][:];
    obsvalue = ds.variables[varname][:];

    ds.close()

    obstime = np.datetime64('1900-01-01T00:00:00') + np.array(24*60*60*obstime.data,dtype=np.int)
    # https://stackoverflow.com/a/26895491/3801401

    sel = (obsdepth < 10) & (lon[0] < obslon) & (obslon < lon[-1]) & (lat[0] < obslat) & (obslat < lat[-1]) & (np.abs(obsvalue) < 200)


    return obsvalue[sel],obslon[sel],obslat[sel],obsdepth[sel],obstime[sel]


def binanalysis(obslon,obslat,obsdepth,obsvalue,obsinvsigma2,lon,lat,depth, dtype = np.float32,
                sigma2_min = (0.2)**2 # dimensional !!!
):
    i = np.array( np.rint( (obslon - lon[0])/(lon[1]-lon[0])), dtype = np.int64 )
    j = np.array( np.rint( (obslat - lat[0])/(lat[1]-lat[0])), dtype = np.int64 )

    sel = (0 <= i) & (i < len(lon)) & (0 <= j) & (j < len(lat))

    lin = j[sel]*len(lon) + i[sel]

    sz = (len(lat),len(lon))
    length = len(lat) * len(lon)


    msum = np.bincount(lin,weights=obsvalue[sel]*obsinvsigma2[sel],minlength = length).reshape(sz);
    minvsigma2 = np.bincount(lin,weights=obsinvsigma2[sel],minlength = length).reshape(sz);


    #print("min sigma2 ",1/minvsigma2.max())

    # correction factor
    alpha = 1/(minvsigma2 * sigma2_min)
    alpha[alpha > 1] = 1

    minvsigma2 = alpha * minvsigma2
    msum = alpha * msum

    #print("min sigma2 after ",1/minvsigma2.max())


    mmean = np.NaN * np.zeros(sz, dtype = dtype)
    #mmean = np.zeros(sz, dtype = dtype)
    mmean[minvsigma2 > 0] = msum[minvsigma2 > 0] / minvsigma2[minvsigma2 > 0]
    #print("bina",minvsigma2[minvsigma2 > 0].min())

    return mmean, np.array(msum, dtype=dtype), np.array(minvsigma2, dtype=dtype)


# month = 1

# sel = obsmonths == month

# mobsvalue,mobslon,mobslat,mobsdepth,mobstime = (
#     obsvalue[sel],obslon[sel],obslat[sel],obsdepth[sel],obstime[sel])


#obsvalue,obslon,obslat,obsdepth,obstime = loadobs(fname,varname)
#obsmonths = obstime.astype('datetime64[M]').astype(int) % 12 + 1

# month = 1

# sel = obsmonths == month

# mobsvalue,mobslon,mobslat,mobsdepth,mobstime = (
#     obsvalue[sel],obslon[sel],obslat[sel],obsdepth[sel],obstime[sel])



# #plt.plot(obslon[sel],obslat[sel],".")
# #plt.show()

# mmean = binaverage(mobslon,mobslat,mobsvalue,lon,lat)

# mobsinvsigma2 = np.ones(mobsvalue.shape)

# mmean,msum,minvsigma2 = binanalysis(mobslon,mobslat,mobsvalue,mobsinvsigma2,lon,lat)


#plt.pcolor(minvsigma2); plt.colorbar(); plt.show()
#plt.pcolor(mmean, cmap="jet"); plt.colorbar(); plt.show()

def dist(lon1,lat1,lon2,lat2):
    return np.sqrt((lon1 - lon2)**2 + (lat2 - lat1)**2)

def loadobsdata(obsvalue,obslon,obslat,obsdepth,obstime,
                train=True, jitter_std_lon = 0., jitter_std_lat = 0., jitter_std_value = 0.):

    nvar = 6
    sz = (len(lat),len(lon))
    ntime = 12
    meandataval = 15
    meandata = np.ma.array(meandataval * np.ones(sz), mask = np.logical_not(mask))
    obsmonths = obstime.astype('datetime64[M]').astype(int) % 12 + 1

    depthr = np.array([0.,5, 10, 15, 20, 25, 30, 40, 50, 66, 75, 85, 100, 112, 125, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1600, 1750, 1850, 2000])

    def datagen():
        for month in range(1,ntime+1):
            sel = obsmonths == month

            mobsvalue,mobslon,mobslat,mobsdepth,mobstime = (
                obsvalue[sel],obslon[sel],obslat[sel],obsdepth[sel],obstime[sel])

            #plt.plot(obslon[sel],obslat[sel],".")
            #plt.show()

            #if train:
            #    mobslon += jitter_std_lon * np.random.randn(mobslon.shape[0])
            #    mobslat += jitter_std_lat * np.random.randn(mobslat.shape[0])

            mobsinvsigma2 = np.ones(mobsvalue.shape)
            mmean,msum,minvsigma2 = binanalysis(mobslon,mobslat,mobsdepth,mobsvalue - meandataval,mobsinvsigma2,lon,lat,depthr)

            # debug
            #mmean[minvsigma2 == 0] = 0
            #minvsigma2[minvsigma2 != 0] = 1

            #plt.pcolor(minvsigma2); plt.colorbar(); plt.show()
            #plt.pcolor(mmean, cmap="jet"); plt.colorbar(); plt.show()
            x = np.zeros((len(lat),len(lon),nvar),dtype = np.float32)
            x[:,:,0] = msum
            #x[:,:,0] = mmean
            x[:,:,1] = minvsigma2
            x[:,:,2] = lon.reshape(1,len(lon))
            x[:,:,3] = lat.reshape(len(lat),1)
            x[:,:,4] = np.cos(2*math.pi * (month-1) / 12)
            x[:,:,5] = np.sin(2*math.pi * (month-1) / 12)

            #print("range x",x[:,:,0].min(),x[:,:,0].max())

            #x = np.stack((msum,minvsigma2),axis=-1)
            xin = x.copy()

            if train:
                # add noise where we have data
                sel = minvsigma2 > 0
                xin[:,:,0][sel] += jitter_std_value * np.random.randn(np.sum(sel))
                xin[:,:,1][sel] = 1/(1/minvsigma2[sel] + jitter_std_value**2)

                xin[:,:,2] += jitter_std_lon * np.random.randn(len(lat),len(lon))
                xin[:,:,3] += jitter_std_lat * np.random.randn(len(lat),len(lon))


            if train:
                size_gap = 1.5
                min_gap_count = 50
                #cvtrain = 0.2
                #selmask = np.random.rand(sz[0],sz[1]) < cvtrain

                while True:
                    gap_lon = lon[0] + (lon[-1]-lon[0]) * random.random()
                    gap_lat = lat[0] + (lat[-1]-lat[0]) * random.random()
                    dist_gap = dist(x[:,:,2],gap_lon,x[:,:,3],gap_lat)
                    selmask = (dist_gap < size_gap) & (xin[:,:,1] > 0)
                    if np.sum(selmask) >= min_gap_count:
                        break

                #     print("too few obs at location. I try again")

                xin[:,:,0][selmask] = 0
                xin[:,:,1][selmask] = 0

            yield (xin,x[:,:,0:2])

    return datagen,ntime,meandata,nvar


def savesample(fname,batch_m_rec,batch_σ2_rec,meandata,lon,lat,e,ii,offset):
    #print("fname,ii",fname,ii)
    fill_value = -9999.
    recdata = batch_m_rec # + meandata;
    batch_sigma_rec = np.sqrt(batch_σ2_rec)

    if ii == 0:
        # create file
        root_grp = Dataset(fname, 'w', format='NETCDF4')

        # dimensions
        root_grp.createDimension('time', None)
        root_grp.createDimension('lon', len(lon))
        root_grp.createDimension('lat', len(lat))

        # variables
        #time = root_grp.createVariable('time', 'f8', ('time',))
        nc_lon = root_grp.createVariable('lon', 'f4', ('lon',))
        nc_lat = root_grp.createVariable('lat', 'f4', ('lat',))
        nc_meandata = root_grp.createVariable('meandata', 'f4', ('lat','lon'), fill_value=fill_value)

        nc_batch_m_rec = root_grp.createVariable('batch_m_rec', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value)
        nc_batch_sigma_rec = root_grp.createVariable('batch_sigma_rec', 'f4', ('time', 'lat', 'lon',), fill_value=fill_value)

        # data
        nc_lon[:] = lon
        nc_lat[:] = lat
        nc_meandata[:,:] = meandata
    else:
        # append to file
        root_grp = Dataset(fname, 'a')
        nc_batch_m_rec = root_grp.variables['batch_m_rec']
        nc_batch_sigma_rec = root_grp.variables['batch_sigma_rec']

    #print("write offset ",offset,batch_m_rec.shape[0] + offset-1)
    for n in range(batch_m_rec.shape[0]):
        nc_batch_m_rec[n+offset,:,:] = np.ma.masked_array(batch_m_rec[n,:,:],meandata.mask) + meandata
        nc_batch_sigma_rec[n+offset,:,:] = np.ma.masked_array(batch_sigma_rec[n,:,:],meandata.mask)


    root_grp.close()


def monthlyCVRMS(lon,lat,depth,value,obsvalue,obslon,obslat,obsdepth,obstime):
    obsmonths = obstime.astype('datetime64[M]').astype(int) % 12


    RMS = np.zeros((12))

    for month in range(0,12):
        sel = obsmonths == month

        mobsvalue,mobslon,mobslat,mobsdepth,mobstime = (
            obsvalue[sel],obslon[sel],obslat[sel],obsdepth[sel],obstime[sel])

        mclim = scipy.interpolate.interpn((depth,lat,lon),value[month,:,:,:],  np.vstack( (mobsdepth,mobslat,mobslon) ).T )
        sel = np.isfinite(mclim);
        RMS[month] = np.sqrt(np.mean((mclim[sel] - mobsvalue[sel])**2))

    totRMS = np.sqrt(np.mean(RMS**2))
    return RMS,totRMS


def monthlyCVRMS_files(fname,fnamecv,varname):
    ds = Dataset(fname);
    lon = ds.variables["lon"][:]
    lat = ds.variables["lat"][:]
    depth = ds.variables["depth"][:]

    #v = ds.variables["mean_rec"][:].filled(np.NaN)
    v = ds.variables[varname][:].filled(np.NaN)

    obsvalue,obslon,obslat,obsdepth,obstime = loadobs(fnamecv,varname)
    ds.close()
    return monthlyCVRMS(lon,lat,depth,v,obsvalue,obslon,obslat,obsdepth,obstime)

jitter_std_lon = 2*(lon[1]-lon[0])
jitter_std_lat = 2*(lat[1]-lat[0])
jitter_std_lon = 0*(lon[1]-lon[0])
jitter_std_lat = 0*(lat[1]-lat[0])

jitter_std_lat = 0.1
jitter_std_lon = jitter_std_lat / math.cos(lat.mean()*math.pi/180)

jitter_std_value = 0.5

# train_datagen,train_len,meandata,nvar = loadobsdata(
#     train = True,
#     jitter_std_lon = jitter_std_lon,
#     jitter_std_lat = jitter_std_lat,
#     jitter_std_value = jitter_std_value)

# test_datagen,test_len,meandata_test,nvar_test = loadobsdata(train = False)


# mask = meandata.mask

# xin,x = next(train_datagen())

# print("xin.shape",xin.shape)
# print("x.shape",x.shape)

# nvar = reconstruct_params["nvar"]
# for i in range(0,nvar):
#     print("range xin",i,xin[:,:,i].min(),xin[:,:,i].max())


# print(xin[2,1,:])
# print(xin[2,2,:])

# it = train_datagen()
# xin,x = next(it)
# #xin,x = next(it)
# #xin,x = next(it)

# #plt.figure(); plt.pcolor(mmean, cmap="jet"); plt.colorbar(); plt.show()
# #plt.figure(); plt.pcolor(xin[:,:,1], cmap="jet"); plt.colorbar(); plt.show()


# #xin,x = next(test_datagen())


default_parameters = [0.1, 4, 1.5]
dimensions = [
    Real(low=1e-6, high=2., prior="log-uniform", name="regularization_L2_beta"),
    #Integer(low=2, high = 6, name = "ndepth"),
    Integer(low=4, high = 6, name = "ndepth"),
    #Real(low=1.1, high = 1.8, name = "ksize_factor")
    Real(low=1., high = 1.5, name = "ksize_factor")
]


def check(regularization_L2_beta,ndepth,ksize_factor):

    #ndepth = 4
    #ksize_factor = 3/2

    enc_ksize_internal = [ int(16 * (ksize_factor)**i) for i in range(ndepth)]

    obsvalue,obslon,obslat,obsdepth,obstime = loadobs(fnametrain,varname)

    train_datagen,train_len,meandata,nvar = loadobsdata(
        obsvalue,obslon,obslat,obsdepth,obstime,
        train = True,
        jitter_std_lon = jitter_std_lon,
        jitter_std_lat = jitter_std_lat,
        jitter_std_value = jitter_std_value)

    test_datagen,test_len,meandata_test,nvar_test = loadobsdata(obsvalue,obslon,obslat,obsdepth,obstime,train = False)

    mask = meandata.mask

    fname = DINCAE.reconstruct(
        lon,lat,mask,meandata,
        train_datagen,train_len,
        test_datagen,test_len,
        outdir,
        **{**reconstruct_params,
           # number of input variables
           "nvar": nvar,
           "regularization_L2_beta": regularization_L2_beta,
           "enc_ksize_internal": enc_ksize_internal
        })

    return fname



@use_named_args(dimensions=dimensions)
def fitness(regularization_L2_beta,ndepth,ksize_factor):

    bestRMS = getattr(fitness, 'bestRMS', 1e9)
    # https://github.com/tensorflow/tensorflow/issues/17048#issuecomment-368082470
    with Pool(1) as p:
        fname = p.apply(check,(regularization_L2_beta,ndepth,ksize_factor))

    fnamecv = os.path.join(basedir,"Temperature.cv.nc")
    varname = "Temperature"

    RMS,totRMS = monthlyCVRMS_files(fname,fnamecv,varname)

    with open(os.path.join(outdir,"test.jsonl"),mode="a") as f:
        data = {'totRMS': totRMS,
                'regularization_L2_beta': regularization_L2_beta,
                'ndepth': int(ndepth),
                'ksize_factor': ksize_factor,
                'RMS': list(RMS)}
        print(data)
        print(json.dumps(data, sort_keys=True, indent=None),
              file=f,flush=True)

    if totRMS < bestRMS:
        fitness.bestRMS = totRMS
    else:
        # remove reconstruction
        os.remove(fname)

    return totRMS

# search_result = skopt.gp_minimize(
#     func=fitness,
#     dimensions=dimensions,
#     acq_func='EI', # Expected Improvement.
#     n_calls=130,
#     x0=default_parameters)

# print("search_result ",search_result)

# print("search_result x",search_result.x)
# print("search_result fun",search_result.fun)

check(0.1, 4, 1.5)
#fname = "/mnt/data1/abarth/work/Data/DINCAE_insitu/Test-jitter-lonlat-0.1-only-input-gap-jitter-smaller-l2-0.7/data-2019-07-08T143513.nc";

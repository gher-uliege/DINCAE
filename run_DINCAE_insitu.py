#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
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

checkmethod = "DINCAE"
#checkmethod = "DIVAnd"
checkmethod = sys.argv[1]

print("checkmethod ",checkmethod)
if checkmethod == "DIVAnd":
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    jl.eval('push!(LOAD_PATH,joinpath(ENV["HOME"],"projects/Julia/share"))')
    jl.eval('push!(LOAD_PATH,joinpath(ENV["HOME"],"src","CAE"))')
    from julia import dincae_insitu


epochs = 5000*2
epochs = 300
#epochs = 50
#epochs = 5
#epochs = 1

reconstruct_params = {
    #"epochs": 1,
    #"epochs": 1_000 * 5 * 2,
    "epochs": epochs,
    #"epochs": 5,
    "batch_size": 12,
    "skipconnections": [],
    #"save_each": 100 * 2,
    #"save_each": 5,
    #"save_each": 20,
    "save_each": 0,
    "dropout_rate_train": 0.3,
    "shuffle_buffer_size": 120,
    #"shuffle_buffer_size": 12,
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

outdir = os.path.join(basedir,"Optimization-" + checkmethod + "-decay-3")


maskname = os.path.join(basedir,"mask.nc")

ds = Dataset(maskname, 'r')
lon = ds.variables["lon"][:].data;
lat = ds.variables["lat"][:].data;
depthr = ds.variables["depth"][:].data;
mask = np.array(ds.variables["mask"][:,:,:].data,dtype = np.bool);

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

    #sel = (obsdepth < 10) & (lon[0] < obslon) & (obslon < lon[-1]) & (lat[0] < obslat) & (obslat < lat[-1]) & (np.abs(obsvalue) < 200)
    sel = (lon[0] < obslon) & (obslon < lon[-1]) & (lat[0] < obslat) & (obslat < lat[-1]) & (np.abs(obsvalue) < 200) & np.isfinite(obsvalue)
    #print("depth ",obsdepth.min(), obsdepth.max())

    return obsvalue[sel],obslon[sel],obslat[sel],obsdepth[sel],obstime[sel]


def binanalysis(obslon,obslat,obsdepth,obsvalue,obsinvsigma2,lon,lat,depth, dtype = np.float32,
                sigma2_min = (0.2)**2 # dimensional !!!
):
    i = np.array( np.rint( (obslon - lon[0])/(lon[1]-lon[0])), dtype = np.int64 )
    j = np.array( np.rint( (obslat - lat[0])/(lat[1]-lat[0])), dtype = np.int64 )
    k = -np.ones(len(obsdepth),dtype = np.int64)

    depthbounds = np.zeros(len(depth)+1)
    depthbounds[0] = depth[0]
    depthbounds[1:-1] = (depth[1:] + depth[0:-1])/2
    depthbounds[-1] = depth[-1]

    for l in range(len(depth)):
        sel =  (depthbounds[l] <= obsdepth) & (obsdepth < depthbounds[l+1])
        #print("sel ",np.sum(sel))
        k[sel] = l

    sel = (0 <= i) & (i < len(lon)) & (0 <= j) & (j < len(lat)) & (k != -1)

    lin = k[sel]*len(lon)*len(lat) + j[sel]*len(lon) + i[sel]

    sz = (len(depth),len(lat),len(lon))
    length = len(depth) * len(lat) * len(lon)

    msum = np.bincount(lin,weights=obsvalue[sel]*obsinvsigma2[sel],minlength = length).reshape(sz);
    minvsigma2 = np.bincount(lin,weights=obsinvsigma2[sel],minlength = length).reshape(sz);


    #print("min sigma2 ",1/minvsigma2.max())

    # correction factor
    alpha = np.ones(msum.shape)
    seldata = minvsigma2 > 0
    alpha[seldata] = 1/(minvsigma2[seldata] * sigma2_min)
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


# mobsinvsigma2 = np.ones(mobsvalue.shape)

# mmean,msum,minvsigma2 = binanalysis(mobslon,mobslat,mobsvalue,mobsinvsigma2,lon,lat)


#plt.pcolor(minvsigma2); plt.colorbar(); plt.show()
#plt.pcolor(mmean, cmap="jet"); plt.colorbar(); plt.show()

def dist(lon1,lat1,lon2,lat2):
    return np.sqrt((lon1 - lon2)**2 + (lat2 - lat1)**2)

def loadobsdata(obsvalue,obslon,obslat,obsdepth,obstime,
                train=True, jitter_std_lon = 0., jitter_std_lat = 0., jitter_std_value = 0.):

    #nvar = 6
    nvar = 11
    #nvar = 7
    sz = (len(lat),len(lon))
    ntime = 12
    meandataval = 15
    meandata = np.ma.array(meandataval * np.ones(sz), mask = np.logical_not(mask[0,:,:]))
    obsmonths = obstime.astype('datetime64[M]').astype(int) % 12 + 1

    nslices = ntime * len(depthr)

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

            for k in range(len(depthr)):
                # debug
                #mmean[minvsigma2 == 0] = 0
                #minvsigma2[minvsigma2 != 0] = 1

                #plt.pcolor(minvsigma2); plt.colorbar(); plt.show()
                #plt.pcolor(mmean, cmap="jet"); plt.colorbar(); plt.show()
                x = np.zeros((len(lat),len(lon),nvar),dtype = np.float32)
                x[:,:,0] = msum[k,:,:]
                #x[:,:,0] = mmean[k,:,:]
                x[:,:,1] = minvsigma2[k,:,:]

                x[:,:,2] = lon.reshape(1,len(lon))
                x[:,:,3] = lat.reshape(len(lat),1)
                x[:,:,4] = depthr[k]
                x[:,:,5] = np.cos(2*math.pi * (month-1) / 12)
                x[:,:,6] = np.sin(2*math.pi * (month-1) / 12)

                # previous layer
                kp = max(k-1,0)
                x[:,:,7] = msum[kp,:,:]
                x[:,:,8] = minvsigma2[kp,:,:]

                kn = min(k+1,len(depthr)-1)
                x[:,:,9] = msum[kn,:,:]
                x[:,:,10] = minvsigma2[kn,:,:]

                #print("range x",k,x.min(),x.max(), np.sum(np.isinf(x)), np.sum(np.isnan(x)) )

                xin = x.copy()

                if train:
                    # add noise where we have data
                    sel = minvsigma2[k,:,:] > 0
                    xin[:,:,0][sel] += jitter_std_value * np.random.randn(np.sum(sel))
                    xin[:,:,1][sel] = 1/(1/minvsigma2[k,:,:][sel] + jitter_std_value**2)

                    xin[:,:,2] += jitter_std_lon * np.random.randn(len(lat),len(lon))
                    xin[:,:,3] += jitter_std_lat * np.random.randn(len(lat),len(lon))


                if train:
                    size_gap = 1.5
                    min_gap_count = 50
                    cvtrain = 0.2
                    selmask = np.random.rand(sz[0],sz[1]) < cvtrain

                    # while True:
                    #     gap_lon = lon[0] + (lon[-1]-lon[0]) * random.random()
                    #     gap_lat = lat[0] + (lat[-1]-lat[0]) * random.random()
                    #     dist_gap = dist(x[:,:,2],gap_lon,x[:,:,3],gap_lat)
                    #     selmask = (dist_gap < size_gap) & (xin[:,:,1] > 0)
                    #     if np.sum(selmask) >= min_gap_count:
                    #         break

                    #     print("too few obs at location. I try again")

                    xin[:,:,0][selmask] = 0
                    xin[:,:,1][selmask] = 0

                yield (xin,x[:,:,0:2])

    return datagen,nslices,meandata,nvar

class Saver4D:
    def __init__(self,depth,mask,varname):
        self.depth = depth
        self.mask = mask
        self.varname = varname

    def __call__(self,fname,m_rec,σ2_rec,meandata,lon,lat,e,ii,offset):
        fill_value = -9999.
        recdata = m_rec + meandata;
        sigma_rec = np.sqrt(σ2_rec)

        if ii == 0:
            # create file
            root_grp = Dataset(fname, "w", format="NETCDF4")

            # dimensions
            root_grp.createDimension("time", None)
            root_grp.createDimension("lon", len(lon))
            root_grp.createDimension("lat", len(lat))
            root_grp.createDimension("depth", len(self.depth))

            # variables
            nc_lon = root_grp.createVariable("lon", "f8", ("lon",))
            nc_lat = root_grp.createVariable("lat", "f8", ("lat",))
            nc_depth = root_grp.createVariable("depth", "f8", ("depth",))

            nc_meandata = root_grp.createVariable(
                "meandata", "f4", ("lat","lon"),
                fill_value=fill_value)

            nc_mean_rec = root_grp.createVariable(
                self.varname, "f4", ("time", "depth", "lat", "lon"),
                fill_value=fill_value)

            nc_sigma_rec = root_grp.createVariable(
                self.varname + "_err", "f4", ("time", "depth", "lat", "lon",),
                fill_value=fill_value)

            # data
            nc_lon[:] = lon
            nc_lat[:] = lat
            nc_depth[:] = self.depth
            nc_meandata[:,:] = meandata
        else:
            # append to file
            root_grp = Dataset(fname, "a")
            nc_mean_rec = root_grp.variables[self.varname]
            nc_sigma_rec = root_grp.variables[self.varname + "_err"]

        for l in range(m_rec.shape[0]):
            # unravel the dimensions
            k = (l+offset) % len(self.depth)
            n = (l+offset) // len(self.depth) # integer division
            #print("l+o,k,n",l+offset,k,n)

            nc_mean_rec[n,k,:,:] = np.ma.masked_array(
                recdata[l,:,:],np.logical_not(self.mask[k,:,:]))
            nc_sigma_rec[n,k,:,:] = np.ma.masked_array(
                sigma_rec[l,:,:],np.logical_not(self.mask[k,:,:]))

        root_grp.close()

        # ntime = 12
        # if offset + m_rec.shape[0] == len(self.depth) * ntime:
        #     fnamecv = os.path.join(basedir,"Temperature.cv.nc")
        #     RMS,totRMS = monthlyCVRMS_files(fname,fnamecv,self.varname)
        #     print("cv totRMS ",totRMS)


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

    #mask = meandata.mask

    fname = DINCAE.reconstruct(
        lon,lat,mask[0,:,:],meandata,
        train_datagen,train_len,
        test_datagen,test_len,
        outdir,
        **{**reconstruct_params,
           # number of input variables
           "nvar": nvar,
           "regularization_L2_beta": regularization_L2_beta,
           "enc_ksize_internal": enc_ksize_internal,
           "savesample": Saver4D(depthr,mask,varname)
        })

    return fname


dimensions = [
    Real(low=1e-6, high=1e-2, prior="log-uniform", name="regularization_L2_beta"),
    #Integer(low=2, high = 6, name = "ndepth"),
    Integer(low=4, high = 6, name = "ndepth"),
    #Real(low=1.1, high = 1.8, name = "ksize_factor")
    Real(low=1., high = 1.5, name = "ksize_factor")
]

@use_named_args(dimensions=dimensions)
def DINCAE_fitness(regularization_L2_beta,ndepth,ksize_factor):

    print("parameters ",regularization_L2_beta,ndepth,ksize_factor)
    bestRMS = getattr(DINCAE_fitness, 'bestRMS', 1e9)
    # https://github.com/tensorflow/tensorflow/issues/17048#issuecomment-368082470
    with Pool(1) as p:
        fname = p.apply(check,(regularization_L2_beta,ndepth,ksize_factor))

    fnamecv = os.path.join(basedir,"Temperature.cv.nc")
    varname = "Temperature"

    RMS,totRMS = monthlyCVRMS_files(fname,fnamecv,varname)

    with open(os.path.join(outdir,"DINCAE.jsonl"),mode="a") as f:
        data = {'totRMS': totRMS,
                'regularization_L2_beta': regularization_L2_beta,
                'ndepth': int(ndepth),
                'ksize_factor': ksize_factor,
                'RMS': list(RMS),
                'filename': fname
        }
        print(data)
        print(json.dumps(data, sort_keys=True, indent=None),
              file=f,flush=True)

    if totRMS < bestRMS:
        DINCAE_fitness.bestRMS = totRMS
    else:
        # remove reconstruction
        os.remove(fname)

    return totRMS

def optim_DINCAE():
    regularization_L2_beta = 1.0e-3
    ndepth = 5
    ksize_factor = 1.2544423914213654

    default_parameters = [regularization_L2_beta,ndepth,ksize_factor]

    search_result = skopt.gp_minimize(
        func=DINCAE_fitness,
        dimensions=dimensions,
        ##acq_func='EI', # Expected Improvement.
        n_calls=30,
        x0=default_parameters)

    print("search_result ",search_result)

    print("search_result x",search_result.x)
    print("search_result fun",search_result.fun)


DIVAnd_dimensions = [
    Real(low=100_000., high=300_000., name="hlen"),
    Real(low=0.01, high = 10., name = "epsilon2")
]

@use_named_args(dimensions=DIVAnd_dimensions)
def DIVAnd_fitness(hlen,epsilon2):

    bestRMS = getattr(DIVAnd_fitness, 'bestRMS', 1e9)

    timestr = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    fname = os.path.join(outdir,"DIVAnd-{}.nc".format(timestr))
    dincae_insitu.DIVAnd_check(hlen,epsilon2,fname)

    fnamecv = os.path.join(basedir,"Temperature.cv.nc")
    varname = "Temperature"

    RMS,totRMS = monthlyCVRMS_files(fname,fnamecv,varname)

    with open(os.path.join(outdir,"DIVAnd.jsonl"),mode="a") as f:
        data = {'totRMS': totRMS,
                'hlen': hlen,
                'epsilon2':  epsilon2,
                'RMS': list(RMS),
                'filename': fname
        }
        print(data)
        print(json.dumps(data, sort_keys=True, indent=None),
              file=f,flush=True)

    if totRMS < bestRMS:
        DIVAnd_fitness.bestRMS = totRMS
    else:
        # remove reconstruction
        os.remove(fname)

    return totRMS


def optim_DIVAnd():

    search_result = skopt.gp_minimize(
        func=DIVAnd_fitness,
        dimensions=DIVAnd_dimensions,
        #acq_func='EI', # Expected Improvement.
        n_calls=100,
        x0=[150_000,1.])

    print("search_result ",search_result)

    print("search_result x",search_result.x)
    print("search_result fun",search_result.fun)


if checkmethod == "DIVAnd":
    optim_DIVAnd()
else:
    optim_DINCAE()
#regularization_L2_beta,ndepth,ksize_factor = (0.1, 4, 1.5)

# regularization_L2_beta = 0.0007977075207819413
# ndepth = 5
# ksize_factor = 1.2544423914213654

#check(regularization_L2_beta,ndepth,ksize_factor)


#DINCAE_fitness([regularization_L2_beta,ndepth,ksize_factor])

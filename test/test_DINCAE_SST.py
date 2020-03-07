#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run on CPU!
# CUDA_VISIBLE_DEVICES=-1  python3 test_DINCAE_SST.py

import DINCAE
import tensorflow as tf
import os
import urllib.request
import pytest

@pytest.fixture()
def small_example():
    filename = "avhrr_sub_add_clouds_small.nc"
    varname = "SST"

    if not os.path.isfile(filename):
       urllib.request.urlretrieve("https://dox.ulg.ac.be/index.php/s/b3DWpYysuw6itOz/download", filename)

    yield (filename,varname)


def test_load(small_example):
    filename,varname = small_example
    lon,lat,time,data,missing,mask = DINCAE.load_gridded_nc(filename,varname)

    train_datagen,nvar,train_len,meandata = DINCAE.data_generator(
        lon,lat,time,data,missing)

    (xin,xtrue) = next(train_datagen())



def reference_reconstruct_gridded_nc():
    filename = "avhrr_sub_add_clouds.nc"
    varname = "SST"
    outdir = "temp-result"
    iseed = 12345
    epochs = 1
    loss = []

    if not os.path.isfile(filename):
       urllib.request.urlretrieve("https://dox.ulg.ac.be/index.php/s/C7rwJ9goIRpvEcC/download", filename)

    DINCAE.reconstruct_gridded_nc(filename,varname,outdir,
                                  iseed = iseed,
                                  epochs = epochs,
                                  save_each = 1,
                                  tensorboard = True,
                                  nprefetch = 1,
                                  nepoch_keep_missing = 10,
                                  truth_uncertain = True,
                                  regularization_L2_beta = 0.001,
                                  loss = loss,
    )


    print("Last training loss: {:.30f}".format(loss[-1]))

    refloss = {
        "1.12.0": 1.610045909881591796875000000000,
        "1.15.0": 1.074228763580322265625000000000
        }


    if tf.__version__ in refloss:
        # 1.12 on travis
        # 1.635820031166076660156250000000

        #1.15 on travis
        # 0.949220180511474609375000000000

        print("loss equal ",loss[-1] == refloss[tf.__version__])
        assert abs(loss[-1] - refloss[tf.__version__]) < 0.2
    else:
        print("warning: no reference value for version tensorflow " + tf.__version__)
        assert loss[-1] < 2


def test_reconstruct_gridded_nc(small_example):
    filename,varname = small_example
    outdir = "temp-result"
    iseed = 12345
    epochs = 1
    loss = []

    DINCAE.reconstruct_gridded_nc(filename,varname,outdir,
                                  iseed = iseed,
                                  epochs = epochs,
                                  save_each = 1,
                                  tensorboard = True,
                                  nprefetch = 1,
                                  nepoch_keep_missing = 10,
                                  truth_uncertain = True,
                                  regularization_L2_beta = 0.001,
                                  loss = loss,
    )


    print("Last training loss: {:.30f}".format(loss[-1]))
    assert loss[-1] < 18


def test_reconstruct_gridded_files(small_example):
    filename,varname = small_example
    outdir = "temp-result"
    iseed = 12345
    epochs = 5
    loss = []

    data = [{
        "filename": filename,
        "varname": varname,
    }]

    DINCAE.reconstruct_gridded_files(
        data,outdir,
        iseed = iseed,
        epochs = epochs,
        save_each = 5,
        loss = loss,
    )

    print("Last training loss: {:.30f}".format(loss[-1]))
    assert loss[-1] < 8

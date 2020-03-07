#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run on CPU!
# CUDA_VISIBLE_DEVICES=-1  python3 test_DINCAE_SST.py

import DINCAE
import tensorflow as tf
import os
import urllib.request


def test_load():
    filename = "avhrr_sub_add_clouds.nc"
    varname = "SST"

    if not os.path.isfile(filename):
       urllib.request.urlretrieve("https://dox.ulg.ac.be/index.php/s/C7rwJ9goIRpvEcC/download", filename)

    lon,lat,time,data,missing,mask = DINCAE.load_gridded_nc(filename,varname)

    train_datagen,nvar,train_len,meandata = DINCAE.data_generator(
        lon,lat,time,data,missing)

    (xin,xtrue) = next(train_datagen())

def test_SST():
    #resize_method = tf.image.ResizeMethod.BILINEAR
    resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    filename = "avhrr_sub_add_clouds.nc"
    varname = "SST"

    if not os.path.isfile(filename):
       urllib.request.urlretrieve("https://dox.ulg.ac.be/index.php/s/C7rwJ9goIRpvEcC/download", filename)

    outdir = None

    iseed = 12345
    epochs = 1
    loss = []

    DINCAE.reconstruct_gridded_nc(filename,varname,outdir,
                                  resize_method = resize_method,
                                  iseed = iseed,
                                  epochs = epochs,
                                  save_each = 1,
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

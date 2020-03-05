#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run on CPU!
# CUDA_VISIBLE_DEVICES=-1  python3 test_DINCAE_SST.py

import DINCAE
import tensorflow as tf
import os

#resize_method = tf.image.ResizeMethod.BILINEAR
resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

filename = os.path.expanduser("~/tmp/Data/Med/AVHRR/Data/avhrr_sub_add_clouds.nc")
varname = "SST"
outdir = None

iseed = 12345
epochs = 1
loss = []

DINCAE.reconstruct_gridded_nc(filename,varname,outdir,
                              resize_method = resize_method,
                              iseed = iseed,
                              epochs = epochs,
                              save_each = -10000000000,
                              loss = loss,
)


print("Last training loss: {:.30f}".format(loss[-1]))

refloss = {
    "1.12.0": 1.610045909881591796875000000000,
    "1.15.0": 1.074228763580322265625000000000
    }

if tf.__version__ in refloss:
    assert loss[-1] == refloss[tf.__version__]
else:
    print("warning: no reference value for version tensorflow " * tf.__version__)
    assert loss[-1] < 2

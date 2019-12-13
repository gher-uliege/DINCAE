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

assert loss[-1] == 1.610045909881591796875000000000

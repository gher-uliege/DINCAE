#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import DINCAE
import tensorflow as tf
import os

#resize_method = tf.image.ResizeMethod.BILINEAR
resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

filename = os.path.expanduser("~/tmp/Data/Med/AVHRR/Data/avhrr_sub_add_clouds.nc")
varname = "SST"
outdir = os.path.expanduser("~/tmp/Data/Med/AVHRR/Fig-jitter-more-skip-avg-pool-nn-mv")

iseed = 12345
epochs = 1
loss = []

DINCAE.reconstruct_gridded_nc(filename,varname,outdir,
                              resize_method = resize_method,
                              iseed = iseed,
                              epochs = epochs,
                              save_each = -10000000000
)

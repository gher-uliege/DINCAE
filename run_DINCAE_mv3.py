#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import DINCAE
import numpy as np
import os

basedir = os.path.expanduser("~/Data/DINCAE-multivariate/Adriatic2/")

data = [
    {
        "filename": os.path.join(basedir,"modis_sst_revlat_add_clouds.nc"),
        "varname":  "sst_t",
        "transfun": (DINCAE.identity, DINCAE.identity)
    },
    {

        "filename": os.path.join(basedir,"color_revlat.nc"),
        "varname": "chlor_a",
        "transfun": (np.log, np.exp),
    },
    {
        "filename": os.path.join(basedir,"CCMP_Wind_Analysis_Adriatic_revlat.nc"),
        "varname":  "uwnd",
        "transfun": (DINCAE.identity, DINCAE.identity)
    },
    {
        "filename": os.path.join(basedir,"CCMP_Wind_Analysis_Adriatic_revlat.nc"),
        "varname":  "vwnd",
        "transfun": (DINCAE.identity, DINCAE.identity)
    },
    ]

outdir = os.path.join(basedir,"test-sst-chlor_a-wind-2-decay-lr-10val")

DINCAE.reconstruct_gridded_files(
    data,outdir,
    learning_rate_decay_epoch = 68.9)

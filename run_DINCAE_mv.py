#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import DINCAE
import numpy as np

filename = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/Adriatic2/color_revlat_add_clouds.nc"
varname = "chlor_a"
outdir = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/Adriatic2//test4/"

transfun = (lambda x: x, lambda x: x)
transfun = (np.log, np.exp)


filename = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/Adriatic2/modis_sst_revlat_add_clouds.nc"
varname = "sst"
outdir = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/Adriatic2//test-sst/"

transfun = (DINCAE.identity, DINCAE.identity)

DINCAE.reconstruct_gridded_nc(
    filename,varname,outdir,
    transfun = transfun
)

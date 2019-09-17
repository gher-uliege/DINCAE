#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import DINCAE

filename = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/Adriatic2/color.nc"
varname = "chlor_a"
outdir = "/media/abarth/03489298-6387-4283-a0f5-9e7152600acc/abarth/Data/DINCAE-multivariate/Adriatic2//test/"

DINCAE.reconstruct_gridded_nc(filename,varname,outdir)

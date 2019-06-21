#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import DINCAE

filename = "/path/to/file.nc"
varname = "SST"
outdir = "/path/to/output/dir"

DINCAE.reconstruct_gridded_nc(filename,varname,outdir)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os

from multiprocessing import Pool
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# jl.eval('push!(LOAD_PATH,joinpath(ENV["HOME"],"projects/Julia/share"))')
# jl.eval('push!(LOAD_PATH,joinpath(ENV["HOME"],"src","CAE"))')
# from julia import dincae_utils

import subprocess

basedir = os.path.expanduser("~/Data/DINCAE-multivariate/Adriatic2/")

data = [
    {
        "filename": os.path.join(basedir,"modis_sst_revlat_add_clouds.nc"),
        "varname":  "sst_t",
    },
    {

        "filename": os.path.join(basedir,"color_revlat_log.nc"),
        "varname": "chlor_a",
    },
    {
        "filename": os.path.join(basedir,"CCMP_Wind_Analysis_Adriatic_revlat.nc"),
        "varname":  "uwnd",
    },
    {
        "filename": os.path.join(basedir,"CCMP_Wind_Analysis_Adriatic_revlat.nc"),
        "varname":  "vwnd",
    },
    ]


kwargs = {
    "epochs": 400,
    "learning_rate_decay_epoch": 75,
}


#for data_case in [data[0:1],data]:
#for data_case in [data]:
for data_case in [data[0:2]]:
    with Pool(1) as p:
        import DINCAE
        outdir = os.path.join(basedir, "-".join([d["varname"] for d in data_case]) + "-" + ("-".join([k+str(v) for (k,v) in kwargs.items()])))
        print("outdir ",outdir)
        fname = p.apply(DINCAE.reconstruct_gridded_files,(data,outdir),kwargs)
        print("reconstruction done ")



        print("outside pool")
        prevdir = os.getcwd()
        os.chdir(outdir)
        varname = data_case[0].get("varname_postprocess",data_case[0]["varname"])
        print("start post process")
        subprocess.run(["julia", "--eval", 'using dincae_utils; dincae_utils.post_process("' + varname + '")'])
        #dincae_utils.post_process(varname)
        print("end post process")
        os.chdir(prevdir)

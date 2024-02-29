#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 17:48:18 2021

@author: santiagogimenez
"""

import xarray as xr

#load CHIRPS v2 files

dataPath = f"{workspace}tesis/datos/crudos/netcdf/"
datosChirps2017 = xr.open_dataset(dataPath+'chirps-v2.0.2007.days_p05.nc', engine="netcdf4")

datosChirps2017.data_vars
datosChirps2017.precip[0].plot()
datosChirps2017.precip.data

datosChirps2017.sel(time=slice("2007-01","2007-03"))
datosChirps2017.longitude

arrayLatitudes = datosChirps2017.la
datosChirps2017.precip.time
datosChirpsMedia2017 = datosChirps2017.precip.mean(dim='time',skipna=True,keep_attrs=True)
datosChirpsMedia2017.plot()



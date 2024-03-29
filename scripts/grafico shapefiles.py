# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:52:45 2021

@author: santi
"""

import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gp
import netCDF4 as nc


workPath = rf"{workspace}Tesis"
cuencaSalado = rf"{workspace}Tesis/Datos/shapes/Cuenca-SaladoStaFe/Cuenca-SaladoStaFe.shp"
cuencaJuramento = rf"{workspace}Tesis/Datos/shapes/Cuenca-Juramento/JURAMENTO.shp"
cuencaBajosSubmeridionales = rf"{workspace}Tesis/Datos/shapes/Cuenca-BajosSubmeridionales/Bajos_Submeridionales.shp"
#data = workPath + r"/Datos/NetCDF/2007.nc"

sns.set(style="whitegrid", palette="pastel", color_codes=True)
sns.mpl.rc("figure", figsize=(10,6))

sf = shp.Reader(cuencaBajosSubmeridionales)

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

df = read_shapefile(sf)
df.shape
df.sample(1)

def plot_shape(id, s=None):
    """ PLOTS A SINGLE SHAPE """
    plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')
    shape_ex = sf.shape(id)
    x_lon = np.zeros((len(shape_ex.points),1))
    y_lat = np.zeros((len(shape_ex.points),1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]
    plt.plot(x_lon,y_lat) 
    x0 = np.mean(x_lon)
    y0 = np.mean(y_lat)
    plt.text(x0, y0, s, fontsize=10)
    # use bbox (bounding box) to set plot limits
    plt.xlim(shape_ex.bbox[0],shape_ex.bbox[2])
    return x0, y0

nombre = 'CUENCA DEL RIO PASAJE O SALADO'
nom_id = int(df[df.NOMBRE == nombre].index.values)
plot_shape(nom_id, nombre)

def plot_map(sf, x_lim = None, y_lim = None, figsize = (11,9)):
    '''
    Plot map with lim coordinates
    '''
    plt.figure(figsize = figsize)
    id=0
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k')
        
        if (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=10)
        id = id+1
    
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)

plot_map(sf)

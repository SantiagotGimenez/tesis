

from shapely.geometry import mapping
import geopandas as gpd
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
import xarray as xr
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os


def calculate_kmeans_clusters(dataset, var='precip', time_var='time', longitude_var='longitude', latitude_var='latitude'):
    """
    Calcula los clústeres utilizando el algoritmo k-means y muestra el gráfico de codo (elbow plot).

    Args:
        dataset (xarray.Dataset): El Dataset que contiene los datos.
        var (str, optional): El nombre de la variable a utilizar para el clustering. Por defecto es 'precip'.
        time_var (str, optional): El nombre de la variable de tiempo en el Dataset. Por defecto es 'time'.
        longitude_var (str, optional): El nombre de la variable de longitud en el Dataset. Por defecto es 'longitude'.
        latitude_var (str, optional): El nombre de la variable de latitud en el Dataset. Por defecto es 'latitude'.

    Returns:
        None

    """
    da = dataset[var].mean(dim=time_var).stack(lat_lon=(latitude_var, longitude_var)).reset_index('lat_lon')
    df = da.to_dataframe()[[latitude_var, longitude_var, var]]
    X = df[[latitude_var, longitude_var]].astype('object')
    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2, 10))
    visualizer.fit(X)
    visualizer.show()



def add_kmeans_labels(dataset, 
                      num_clusters=5, 
                      precip_var='precip', 
                      time_var='time', 
                      latitude_var='latitude', 
                      longitude_var='longitude', 
                      ):
    """
    Agrega las etiquetas de clústeres k-means al Dataset dado.

    Args:
        dataset (xarray.Dataset): El Dataset original.
        num_clusters (int, optional): El número de clústeres a generar. Por defecto es 5.
        precip_var (str, optional): El nombre de la variable de precipitación en el Dataset. Por defecto es 'precip'.
        time_var (str, optional): El nombre de la variable de tiempo en el Dataset. Por defecto es 'time'.
        latitude_var (str, optional): El nombre de la variable de latitud en el Dataset. Por defecto es 'latitude'.
        longitude_var (str, optional): El nombre de la variable de longitud en el Dataset. Por defecto es 'longitude'.

    Returns:
        xarray.Dataset: El nuevo Dataset con una nueva variable llamada 'labels' que contiene las etiquetas de clústeres k-means.

    """
    # Extraer la variable de precipitación y las coordenadas
    precip = dataset[precip_var]
    time = dataset[time_var]
    latitude = dataset[latitude_var]
    longitude = dataset[longitude_var]
    # Convertir la variable de precipitación en una matriz 2D
    precip_2d = precip.mean(dim=time_var).stack(points=[latitude_var, longitude_var]).values.reshape(-1,1)
    # Crear el objeto del modelo de k-means
    kmeans = KMeans(n_clusters=num_clusters)
    # Ajustar el modelo de k-means a los datos de precipitación
    kmeans.fit(precip_2d)
    # Obtener las etiquetas de cluster asignadas a cada punto de datos
    cluster_labels = kmeans.labels_
    # Obtener las coordenadas de tiempo, latitud y longitud
    coords = {latitude_var: latitude, longitude_var: longitude}
    # Crear un nuevo DataArray con las etiquetas y las mismas coordenadas que precip
    labels_dataarray = xr.DataArray(cluster_labels.reshape(len(latitude), len(longitude)), dims=[latitude_var, longitude_var], coords=coords)
    # Agregar el nuevo DataArray como una nueva variable 'labels' al Dataset original
    dataset_with_labels = dataset.assign(labels=labels_dataarray)


    return dataset_with_labels


def crop_dataset(dataset, shapefile, mswep=False):
    """
    Recorta un Dataset utilizando una geometría y devuelve el Dataset recortado.

    Args:
        dataset (xarray.Dataset): El Dataset a recortar.
        shapefile (geopandas.GeoDataFrame): La geometría utilizada para recortar el Dataset.
        mswep (bool, optional): Indica si el Dataset es del tipo MSWEP. Por defecto es False.

    Returns:
        xarray.Dataset: El Dataset recortado.

    """
    dataset.rio.write_crs("epsg:4326", inplace=True)
    if mswep:
        dataset.rio.set_spatial_dims("lon", "lat", inplace=True)
    dataset_cropped = dataset.rio.clip(shapefile.geometry.apply(mapping), shapefile.crs, drop=False)
    return dataset_cropped


def max_lag_plot(dataset, shapefile, longitude_dim='longitude', latitude_dim='latitude'):
    """
    Genera un gráfico de la correlación promedio en función del lag y resalta el máximo lag.

    Args:
        dataset (xarray.DataArray): El arreglo de correlaciones llenado.
        shapefile (geopandas.GeoDataFrame): La geometría de la cuenca de Juramento.
        longitude_dim (string): Nombre de la dimensión longitud.
        latitude_dim (string): Nombre de la dimensión latitud.

    Returns:
        None

    """
    correlation_data_crop = crop_dataset(dataset, shapefile)

    df = pd.DataFrame({'lag': correlation_data_crop.mean(dim=longitude_dim).mean(dim=latitude_dim)})
    max_lag = df[df['lag'] == df['lag'].max()].index
    print('El máximo lag es:', max_lag.values)

    ax = plt.subplot()
    correlation_data_crop.mean(dim=longitude_dim).mean(dim=latitude_dim).plot(ax=ax)
    ax.axvline(x=max_lag, color='red', linestyle='--')
    ax.grid()


def add_lagged_columns(dataframe, lag_values):
    """
    Lag the columns in the DataFrame based on the specified lag values and add them as new columns.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the columns to be lagged.
        lag_values (list): The list of lag values to apply.

    Returns:
        pd.DataFrame: The DataFrame with lagged columns added.

    """
    lagged_dataframe = dataframe.copy()
    for column in dataframe.columns:
        for lag in lag_values:
            lagged_column = dataframe[column].shift(lag)
            lagged_dataframe[f"{column}_lag_{lag}"] = lagged_column

    return lagged_dataframe
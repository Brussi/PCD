import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from pathlib import Path
import geopandas as gpd

def read_files():
    #latlon_file="estacao_lat_lon.pkl"
    pcd_file="df_7days.xlsx"
    dir_ = Path("/mnt/e/Cemaden_Bolsista/4cemaden/GeoHydroRiskMap/PCD_data")

    #df_latlon = pd.read_pickle(dir_/latlon_file)
    df_7days = pd.read_excel(dir_/pcd_file)
    return df_7days


def get_points(lower_resol_factor=1, x_limit = -34.6, mask_data = True):
    import xarray as xr
    import numpy as np

    fname = '../data/thresholds/limiares_2023_lower.tif'
    ds = xr.open_rasterio(fname)
    #reduce resolution
    ds = ds.sel(x=slice(ds.x[0], -34.61))
    data = ds[0].values
    x = ds.x[::lower_resol_factor]
    y = ds.y[::lower_resol_factor]
    data = data[::lower_resol_factor, ::lower_resol_factor]
    x, y = np.meshgrid(x, y)

    if mask_data == True:
#        x = x.flatten()
#        y = y.flatten()
        #mask = np.where(ds[0].values.flatten()!=65535)
        mask = np.where(data==65535)
        return x, y, mask
        # x = x[mask]
        # y = y[mask]

    return x, y, mask


def KD(Xf, Yf, Xo, Yo, data, filter_dist=False, nn=4):#f= final grid, o = observed points
    import pyproj
    from scipy.spatial import cKDTree

    p = pyproj.Proj(proj='utm', zone=22, ellps='WGS84', preserve_units=True)
    xA, yA = p(Xf, Yf)
    xB, yB = p(Xo, Yo)
    nA = np.array(list(zip(xA, yA)) )
    nB = np.array(list(zip(xB, yB)) )
 
    btree = cKDTree(nB) 
    dist, idx = btree.query(nA, k=nn)

    if filter_dist == True:
        filter = np.where(dist>100000)
        dist[filter] = np.inf
        #idx = idx[filter] 

    w = 1.0 / dist**2
    data_idw = np.sum(w * data[idx], axis=1) / np.sum(w, axis=1)
    return data_idw#, filter#.reshape(shape_)


def non_PCD_area_idx(xA, yA, xB, yB):#f= final grid, o = observed points
    import pyproj
    from scipy.spatial import cKDTree

    shape = xA.shape
    p = pyproj.Proj(proj='utm', zone=22, ellps='WGS84', preserve_units=True)
    xA, yA = p(xA.flatten(), yA.flatten())
    xB, yB = p(xB, yB)
    nA = np.array(list(zip(xA, yA)))
    nB = np.array(list(zip(xB, yB)))
 
    btree = cKDTree(nB) 
    dist, idx = btree.query(nA, k=1)
    dist = dist.reshape(shape)
    filter = np.where(dist>200000)
        #idx = idx[filter] 
    return filter


def OrdinaryKriging():
    # Example rain gauge data
    df = read_files()
    rainfall_values = df.iloc[:,3:].sum(axis=1)  # Example rainfall values at each gauge

    # Create an instance of OrdinaryKriging
    OK = OrdinaryKriging(
        df.longitude,
        df.latitude,
        rainfall_values,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )

    # Fit the Kriging model
    x, y, mask = get_points()
    z, ss = OK.execute("points", x, y)
    plt.imshow(z)
    plt.show()


def inverse_square_distance():
    df = read_files()
    rainfall_values = df.iloc[:,3:].sum(axis=1)  # Example rainfall values at each gauge
    Xf, Yf, mask_border  = get_points()

    rain_idw = KD(Xf.flatten(), Yf.flatten(), df.longitude, df.latitude, rainfall_values.to_numpy())
    rain_idw = rain_idw.reshape(Xf.shape)
    filter = non_PCD_area_idx(Xf, Yf, df.longitude, df.latitude)

    return rain_idw

def refill():
    a=2

def plot_interpolated(Xf, Yf, var):
    # x = Xf.reshape(1561, 1806)
    # y = Yf.reshape(1561, 1806)
    # var = var.reshape(1561, 1806)
    # m = Basemap(
    #     projection = 'merc',
    #     llcrnrlon= -74,
    #     llcrnrlat= 5,
    #     urcrnrlon= -29,
    #     urcrnrlat= -34, 
    #     resolution= 'i')
    # m.drawcoastlines()
    # x, y = m(x, y)
    # cb = m.contourf(x, y, var)
    # cbar = m.colorbar(cb, location = 'right', pad = '10%')
    #plt.show()
    fig = plt.figure(figsize=(10,5))
    plt.pcolormesh(rain_idw.transpose())
    plt.xlim([0, var.shape[0]])
    plt.ylim([0, var.shape[1]])
    plt.colorbar()
    plt.title("IDW of square distance \n using 10 neighbors")



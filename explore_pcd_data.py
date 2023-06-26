import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd

def read_files():
    #latlon_file="estacao_lat_lon.pkl"
    pcd_file="df_7days.xlsx"
    dir_ = Path("/mnt/e/Cemaden_Bolsista/4cemaden/GeoHydroRiskMap/PCD_data")

    #df_latlon = pd.read_pickle(dir_/latlon_file)
    df_7days = pd.read_excel(dir_/pcd_file)
    return df_7days

def image(df):
    #path = image_file.parent.parent.parent
    fig, ax = plt.subplots(figsize=(15, 15))    
    fig.patch.set_visible(True)    
    fig.suptitle(f'PCDs - Cemaden')   
    plt.scatter(df.longitude, df.latitude, s=10,linewidths=0.2, edgecolor='black')
    #plot states
    shp = "../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    states = gpd.read_file(shp)
    states.boundary.plot(ax=ax,  linewidth=1,edgecolor='black' )

    ax.axis('off')
    #ax.margins(0,0) 
    plt.show()
    #fig.savefig(image_file,dpi=80,bbox_inches='tight',pad_inches=0.6) #pad_inches=-0.2, dpi=200, 

def get_points():
    import xarray as xr
    import numpy as np

    fname = '../data/thresholds/limiares_2023_lower.tif'
    ds = xr.open_rasterio(fname)
    x, y = np.meshgrid(ds.x.values, ds.y.values)
    x = x.flatten()
    y = y.flatten()

    mask = np.where(ds[0].values.flatten()==65535)

    x = x[mask]
    y = y[mask]

    # data_dict = {"longitude": x, "latitude": y}
    # df_grid = pd.DataFrame(data_dict)
    return x, y

def ckdnearest(Xf, Yf, Xo, Yo):#f= final grid, o = observed points
    import pyproj
    from scipy.spatial import cKDTree

    p = pyproj.Proj(proj='utm', zone=22, ellps='WGS84', preserve_units=True)

    xA, yA = p(Xf, Yf)
    xB, yB = p(Xo, Xo)
    nA = np.array(list(zip(xA, yA)) )
    nB = np.array(list(zip(xB, yB)) )
 
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)


    return gdB.iloc[idx,:].reset_index()

def OrdinaryKriging():
    # Example rain gauge data
    df = read_files()
    rainfall_values = df.iloc[:,3:].sum(axis=1)  # Example rainfall values at each gauge

    # Create an instance of OrdinaryKriging
    OK = OrdinaryKriging(
        df.longitude, df.latitude,
        rainfall_values,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )

    # Fit the Kriging model
    x, y = get_points()
    z, ss = OK.execute("points", x, y)
    plt.imshow(z)
    plt.show()
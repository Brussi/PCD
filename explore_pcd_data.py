from matplotlib.image import interpolations_names
import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from pathlib import Path
import geopandas as gpd



def read_files(operational=False):
    dir_ = Path("/mnt/e/Cemaden_Bolsista/4cemaden/GeoHydroRiskMap/PCD_data")

    latlon_file="estacao_lat_lon.pkl"
    df_latlon = pd.read_pickle(dir_/latlon_file)
    pcd_file = Path("2023_06_28-17_22.pkl")

    if pcd_file.suffix == ".pkl":
        df = pd.read_pickle(dir_/pcd_file)
        df = pd.concat([df_latlon, df], axis=1,join='inner')
    if pcd_file.suffix == ".xlsx":
        df = pd.read_excel(dir_/pcd_file)

    if operational == True:
        if "PCD" in os.listdir(dir_):
            df = pd.read_pickle(pcd_file)

    return df, pcd_file.stem


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


def KD(Xf, Yf, Xo, Yo, data, nn=4):#f= final grid, o = observed points
    import pyproj
    from scipy.spatial import cKDTree

    p = pyproj.Proj(proj='utm', zone=22, ellps='WGS84', preserve_units=True)
    xA, yA = p(Xf, Yf)
    xB, yB = p(Xo, Yo)
    nA = np.array(list(zip(xA, yA)) )
    nB = np.array(list(zip(xB, yB)) )
 
    btree = cKDTree(nB) 
    print(nn)
    dist, idx = btree.query(nA, k=nn)

    # if filter_dist == True:
    #     filter = np.where(dist>100000)
    #     dist[filter] = np.inf
        #idx = idx[filter] 

    w = 1.0 / dist**2
    data_idw = np.sum(w * data[idx], axis=1) / np.sum(w, axis=1)
    return data_idw#, filter#.reshape(shape_)


def non_PCD_area_idx(xA, yA, xB, yB, filter_radius):#f= final grid, o = observed points
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
    filter = np.where(dist>filter_radius)
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


def inverse_square_distance(limit_radius=200000, filter=True):
    df, PCD_file = read_files()
    rainfall_values = df.iloc[:,3:].sum(axis=1)  # Example rainfall values at each gauge
    Xf, Yf, mask_border  = get_points()
    #number of neighbou 
    nn=10   
    rain_idw = KD(Xf.flatten(), Yf.flatten(), df.longitude, df.latitude, rainfall_values.to_numpy(), nn)
    rain_idw = rain_idw.reshape(Xf.shape)
    filter = non_PCD_area_idx(Xf, Yf, df.longitude, df.latitude, limit_radius)
    interpol = "inverso da distância ao quadrados"
    if filter == True:
        Xf = Xf[filter]
        Yf = Yf[filter]
        rain_idw = rain_idw[filter]
    else:
        limit_radius = "Sem limite"

    plot(Xf, Yf, 
         rain_idw,
         df,
         limit_radius,
         interpol,
         mask_border,
         nn,
         PCD_file)


def plot(x, y, rain_idw, df, limit_radius, interpol, mask_border, nn, PCD_file):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #map_type = "pcolormesh"
    map_type = "contourf"
    fig, ax = plt.subplots(figsize=(15, 15))    
    fig.patch.set_visible(True)    
    fig.suptitle(f'PCDs - Cemaden- 7 dias acumulados sem decaimento\n'
                 f'Raio limite: {limit_radius}\n'
                 f'Interpolação: {interpol}'
                 f'file: {PCD_file}
    )  

    # draw filled contours.
    clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
            50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750]
    # In future MetPy
    # norm, cmap = ctables.registry.get_with_boundaries('precipitation', clevs)
    cmap_data = [(1.0, 1.0, 1.0),
                (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
                (0.0, 1.0, 1.0),
                (0.0, 0.8784313797950745, 0.501960813999176),
                (0.0, 0.7529411911964417, 0.0),
                (0.501960813999176, 0.8784313797950745, 0.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.6274510025978088, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.125490203499794, 0.501960813999176),
                (0.9411764740943909, 0.250980406999588, 1.0),
                (0.501960813999176, 0.125490203499794, 1.0),
                (0.250980406999588, 0.250980406999588, 1.0),
                (0.125490203499794, 0.125490203499794, 0.501960813999176),
                (0.125490203499794, 0.125490203499794, 0.125490203499794),
                (0.501960813999176, 0.501960813999176, 0.501960813999176),
                (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
                (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
                (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
                (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
                (0.4000000059604645, 0.20000000298023224, 0.0)]
    cmap = mcolors.ListedColormap(cmap_data, 'precipitation')
    norm = mcolors.BoundaryNorm(clevs, cmap.N)

    #cb = plt.tricontourf(Xf.flatten(),Yf.flatten(), rain_idw.flatten())
    rain_idw[mask_border] = np.nan

    if map_type == "contourf":
        cb = plt.contourf(x, y, rain_idw.reshape(x.shape),clevs,cmap=cmap, norm=norm)
    if map_type == "pcolormesh":        
        cb = plt.pcolormesh(x, y, rain_idw.reshape(x.shape), cmap=cmap, norm=norm)


    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)    
    # plt.colorbar(cb, cax=cax, ticks=clevs)
    plt.colorbar(cb, ticks=clevs, shrink=0.5)
    shp = "../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    states = gpd.read_file(shp)

    states.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
    plt.scatter(df.longitude, df.latitude, c=df.iloc[:,3:].sum(axis=1), cmap=cmap, norm=norm, s=0.5, linewidths=0.06, edgecolor='black')
    #plt.show()
    #plt.close()

    ax.axis('off')
    ax.margins(0,0)
    image_file = f"PCD_data/{interpol}_{nn}_limit_radius.png"
    fig.savefig(image_file,dpi=1200,bbox_inches='tight',pad_inches=0.6) #pad_inches=-0.2, dpi=200, 
    print(f"Image generated: {image_file}\n")

inverse_square_distance(
    limit_radius=200000,
    filter=False
    )
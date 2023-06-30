from mpl_toolkits.basemap import Basemap    
import matplotlib.pyplot as plt
import shapefile as shp
import matplotlib.colors as mcolors

def show_points(df):
    #path = image_file.parent.parent.parent
    fig, ax = plt.subplots(figsize=(15, 15))    
    fig.patch.set_visible(True)    
    fig.suptitle(f'PCDs - Cemaden')   
    plt.scatter(df.longitude, df.latitude, s=10,linewidths=0.2, edgecolor='black')
    plt.scatter(x.flatten(), y.flatten(), s=1,linewidths=0.2, edgecolor='black')
   
    #plot states
    shp = "../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    states = gpd.read_file(shp)
    states.boundary.plot(ax=ax,  linewidth=1, edgecolor='black' )

    ax.axis('off')
    #ax.margins(0,0) 
    plt.show()
    #fig.savefig(image_file,dpi=80,bbox_inches='tight',pad_inches=0.6) #pad_inches=-0.2, dpi=200, 

def plot_map(sf, x_lim = None, y_lim = None, figsize = (11,9)):
    plt.figure(figsize = figsize)
    id=0

#calling the function and passing required parameters to plot the full map
def plot(x, y, var):
    fig, ax = plt.subplots(figsize=(15, 15))    
    fig.patch.set_visible(True)    
    fig.suptitle(f'PCDs - Cemaden')  

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
    rain_idw[filter] = np.nan
    cb = plt.contourf(Xf,Yf, rain_idw.reshape(Xf.shape),clevs,cmap=cmap, norm=norm,shading='gouraud')
    #cb = plt.pcolormesh(Xf,Yf, rain_idw.reshape(Xf.shape), cmap=cmap, norm=norm)

    #plt.imshow(rain_idw)
    cbar = m.colorbar(cb, location = 'right', pad = '10%')
    shp = "../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    states = gpd.read_file(shp)
    states.boundary.plot(ax=ax, linewidth=1,edgecolor='black')
    plt.scatter(df.longitude, df.latitude, s=10,linewidths=0.2, edgecolor='black')
    plt.show()
    
def copied():
    import metpy
    from siphon.catalog import TDSCatalog
    from datetime import datetime, timedelta
    %matplotlib inline
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from metpy.units import units
    import metpy.calc as mpcalc
    import matplotlib.pyplot as plt
    import matplotlib.colors as cls
    from xarray.backends import NetCDF4DataStore
    import xarray as xr
    from scipy.ndimage import gaussian_filter
    import numpy as np

    # Set up access via NCSS
    gfs_catalog = ('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                'Global_0p5deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p5deg/Best')
    cat = TDSCatalog(gfs_catalog)
    ncss = cat.datasets[0].subset()
    query3 = ncss.query()
    query3.accept('netcdf')
    query3.variables('Pressure_reduced_to_MSL_msl', 'Precipitation_rate_surface', 'Snow_depth_surface', 'Categorical_Snow_surface')
    now = datetime.utcnow()
    query3.time_range(now, now + timedelta(days=4))
    query3.lonlat_box(west=-140, east=-60, north=60, south=20)
    data3 = ncss.get_data(query3)
    ds3 = xr.open_dataset(NetCDF4DataStore(data3))

    #parsing data
    isSnow_var = ds3.metpy.parse_cf('Categorical_Snow_surface')
    precip_var = ds3.metpy.parse_cf('Precipitation_rate_surface')
    longitude = precip_var.metpy.x
    latitude = precip_var.metpy.y
    time_index = 11

    #All the Precip Stuff
    precip_inch_hour = precip_var[time_index].squeeze() *  141.73228346457 
    precip2 = mpcalc.smooth_n_point(precip_inch_hour, 5, 1)

    #Converting to 10:1 snow ratio
    snow_precip = (precip_inch_hour * isSnow_var[time_index].squeeze()) * 10

    # Plot using CartoPy and Matplotlib
    mapproj = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.)

    # Set projection of data
    data_projection = ccrs.PlateCarree()

    # Grab data for plotting state boundaries
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes',
        scale='50m',
        facecolor='none')

    # Set extent and plot map lines
    fig = plt.figure(1, figsize=(25.,25.))
    ax = plt.subplot(111, projection=mapproj)
    ax.set_extent([-125., -70, 25., 50.], ccrs.PlateCarree())
    ax.coastlines('50m', edgecolor='black', linewidth=0.75)
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)

    #colormap data
    precip_colors = [
    "#bde9bf",  # 0.01 - 0.02 inches 1
    "#adddb0",  # 0.02 - 0.03 inches 2
    "#9ed0a0",  # 0.03 - 0.04 inches 3
    "#8ec491",  # 0.04 - 0.05 inches 4
    "#7fb882",  # 0.05 - 0.06 inches 5
    "#70ac74",  # 0.06 - 0.07 inches 6
    "#60a065",  # 0.07 - 0.08 inches 7
    "#519457",  # 0.08 - 0.09 inches 8
    "#418849",  # 0.09 - 0.10 inches 9
    "#307c3c",  # 0.10 - 0.12 inches 10
    "#1c712e",  # 0.12 - 0.14 inches 11
    "#f7f370",  # 0.14 - 0.16 inches 12
    "#fbdf65",  # 0.16 - 0.18 inches 13
    "#fecb5a",  # 0.18 - 0.2 inches 14
    "#ffb650",  # 0.2 - 0.3 inches 15
    "#ffa146",  # 0.3 - 0.4 inches 16
    "#ff8b3c",   # 0.4 - 0.5 inches 17
    "#ff8b3c"   # 0.5 - 0.6 inches 18
    ]

    precip_colormap = cls.ListedColormap(precip_colors)

    #Precip Rate
    clev_precip =  np.concatenate((np.arange(0.01, 0.1, .01), np.arange(.1, .2, .02), np.arange(.2, .61, .1)))
    norm = cls.BoundaryNorm(clev_precip, 18)
    cf = ax.contourf(longitude, latitude, precip2, clev_precip, cmap=precip_colormap, norm=norm, transform=ccrs.PlateCarree())
    cb = plt.colorbar(cf, ticks=clev_precip, aspect=65, orientation = 'horizontal', shrink=0.6, pad=0.01)

    snow_colors = [
    "#63c9d5",  # 0.1 - 0.2 inches 1
    "#5fb4ca",  # 0.2 - 0.3 inches 2
    "#5a9fc0",  # 0.3 - 0.4 inches 3
    "#558ab5",  # 0.4 - 0.5 inches 4
    "#4e76aa",  # 0.5 - 0.6 inches 5
    "#4763a0",  # 0.6 - 0.7 inches 6
    "#3e4f95",  # 0.7 - 0.8 inches 7
    "#353c8b",  # 0.8 - 0.9 inches 8
    "#292980",  # 0.9 - 1.0 inches 9
    "#493387",  # 1.0 - 1.2 inches 10
    "#643e8e",  # 1.2 - 1.4 inches 11
    "#7c4995",  # 1.4 - 1.6 inches 12
    "#94559c",  # 1.6 - 0.18 inches 13
    "#ab61a3",  # 1.8 - 2 inches 14
    "#c36eaa",  # 2 - 3 inches 15
    "#da7bb0",  # 3 - 4 inches 16
    "#f288b7"   # 4 - 5 inches 17
    ]

    snow_colormap = cls.ListedColormap(snow_colors)

    #Snow Rate
    clev_snow =  np.concatenate((np.arange(.1, 1, .1), np.arange(1, 2, .2), np.arange(2, 6, 1)))
    norm2 = cls.BoundaryNorm(clev_snow, 17)
    cf2 = ax.contourf(longitude, latitude, snow_precip, clev_snow, cmap=snow_colormap, norm=norm2, transform=ccrs.PlateCarree())
    cb2 = plt.colorbar(cf2, ticks=clev_snow, orientation = 'horizontal', pad=0.01, shrink=0.6, aspect=65)

    #Valid Time
    vtime = isSnow_var.metpy.time[time_index].values

    #Title Info
    plt.title('MSLP (hPa) with Highs and Lows, 1000-500 hPa Thickness (m), Rain (in/hr), Snow 10:1 (in/hr)', loc='left')
    plt.title(f'VALID: {vtime}', loc='right')


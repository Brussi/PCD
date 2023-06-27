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
plot_map(sf)
def plot(x, y, var):
    # y_lim = (-35,7) # latitude
    # x_lim = (-75, -33) # longitude
    # shp_path = "../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    # sf = shp.Reader(shp_path)
    # for shape in sf.shapeRecords():
    #     x = [i[0] for i in shape.shape.points[:]]
    #     y = [i[1] for i in shape.shape.points[:]]w
    #     plt.plot(x, y, 'k')
        
    #     if (x_lim == None) & (y_lim == None):
    #         x0 = np.mean(x)
    #         y0 = np.mean(y)
    #         plt.text(x0, y0, id, fontsize=10)
    #     id = id+1
    
    # if (x_lim != None) & (y_lim != None):     
    #     plt.xlim(x_lim)
    #     plt.ylim(y_lim)
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
    cb = plt.contourf(Xf,Yf, rain_idw.reshape(Xf.shape),clevs,cmap=cmap, norm=norm)

    #plt.imshow(rain_idw)
    cbar = m.colorbar(cb, location = 'right', pad = '10%')
    shp = "../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    states = gpd.read_file(shp)
    states.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

    plt.show()

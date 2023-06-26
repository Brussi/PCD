from mpl_toolkits.basemap import Basemap    
import matplotlib.pyplot as plt


def show_points(df):
    #path = image_file.parent.parent.parent
    fig, ax = plt.subplots(figsize=(15, 15))    
    fig.patch.set_visible(True)    
    fig.suptitle(f'PCDs - Cemaden')   
    plt.scatter(df.longitude, df.latitude, s=10,linewidths=0.2, edgecolor='black')
    #plot states
    shp = "../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    states = gpd.read_file(shp)
    states.boundary.plot(ax=ax,  linewidth=1, edgecolor='black' )

    ax.axis('off')
    #ax.margins(0,0) 
    plt.show()
    #fig.savefig(image_file,dpi=80,bbox_inches='tight',pad_inches=0.6) #pad_inches=-0.2, dpi=200, 


def plot(x, y, var):
    m = Basemap(
        projection = 'merc',
        llcrnrlon= -74,
        llcrnrlat= 5,
        urcrnrlon= -29,
        urcrnrlat= -34, 
        resolution= 'i')
    m.drawcoastlines()
    x, y = m(x, y)
    cb = m.contourf([x,y], var)
    cbar = m.colorbar(cb, location = 'right', pad = '10%')
    plt.show()
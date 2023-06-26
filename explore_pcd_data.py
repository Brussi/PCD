import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
from pathlib import Path
import matplotlib.pyplot as plt

def read_files(
        latlon_file="estacao_lat_lon.pkl",
        pcd_file="df_7days.xlsx",
        dir_ = Path("/mnt/e/Cemaden_Bolsista/4cemaden/GeoHydroRiskMap/PCD_data")
        ):


    df_latlon = pd.read_pickle(dir_/latlon_file)
    df_7days = pd.read_excel(dir_/pcd_file)

def image(self, df):
    #path = image_file.parent.parent.parent
    fig, ax = plt.subplots(figsize=(15, 15))    
    fig.patch.set_visible(True)    
    fig.suptitle(f'PCDs - Cemaden')   
    plt.scatter(df.longitude, df.latitude)
    #plot states
    shp = "../../data/shapefiles/br_unidades_da_federacao/BRUFE250GC_SIR.shp"
    states = gpd.read_file(shp)
    states.boundary.plot(ax=ax,  linewidth=1,edgecolor='black' )

    ax.axis('off')
    #ax.margins(0,0) 
    plt.show()
    #fig.savefig(image_file,dpi=80,bbox_inches='tight',pad_inches=0.6) #pad_inches=-0.2, dpi=200, 


# Example rain gauge data
gauge_locations = np.array([[0, 0], [1, 0], [0, 1]])  # Example gauge locations (x, y)
rainfall_values = np.array([10, 5, 7])  # Example rainfall values at each gauge

# Define the target grid or locations where you want to interpolate rainfall
target_locations = np.array([[0.5, 0.5], [0.2, 0.8]])  # Example target locations (x, y)

# Create an instance of OrdinaryKriging
kriging_model = OrdinaryKriging(
    gauge_locations[:, 0], gauge_locations[:, 1], rainfall_values
)

# Fit the Kriging model
kriging_model.krige()

# Perform the Kriging interpolation at the target locations
interpolated_rainfall, _ = kriging_model.predict(target_locations[:, 0], target_locations[:, 1])

print(interpolated_rainfall)
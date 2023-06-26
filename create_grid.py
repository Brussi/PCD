import xarray as xr
import numpy as np

fname = '../data/thresholds/limiares_2023_lower.tif'
ds = xr.open_rasterio(fname)
#x, y = np.meshgrid(ds.x.values, ds.y.values)
# x = x.flatten()
# y = y.flatten()
x = ds.x.values
y = ds.y.values
#mask = np.where(ds[0].values==65535)

# x = x[mask[1]]
# y = y[mask[0]]

data_dict = {"longitude": x, "latitude": y}
df_grid = pd.DataFrame(data_dict)

df_grid.to_pickle("regular_grid.pkl")
import numpy as np
from pykrige.ok import OrdinaryKriging

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

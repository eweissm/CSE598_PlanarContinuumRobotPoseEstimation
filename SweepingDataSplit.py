import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir, Ridge, Input, Output
import matplotlib.pyplot as plt


# Load dataset
file_path = "ExtractedData/ExtractedData_SmoothSine.csv"  # Replace with actual file path
data = pd.read_csv(file_path)

# Extract features and labels
sensor_data = data.iloc[:, 1:7].values  # Columns 2-7 (zero-indexed as 1:7)
coefficients = data.iloc[:, 7:13].values  # Columns 8-13
actuator_inputs = data.iloc[:, 13:15].values  # Columns 14-15

# Concatenate sensor data and actuator inputs as input features
X = np.hstack((sensor_data, actuator_inputs))
y = coefficients  # Ground truth coefficients

DataSplit = np.linspace(.1, .9,10)
mse_vals = []

#Define Echo State Network
n_reservoir = 200  # Number of reservoir neurons
reservoir1 = Reservoir(n_reservoir, name="res1-1")
readout1 = Ridge(ridge=1e-5, name="readout1-1")
model = reservoir1 >> readout1

for i in DataSplit:

    # Split dataset into training (90%) and validation (10%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=i)


    # Train the ESN on training data
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.run(X_val)

    # Compute validation error
    mse = np.mean((y_pred - y_val) ** 2)
    print(f"Validation MSE: {mse:.6f}")
    mse_vals.append(mse)

plt.plot(DataSplit, mse_vals)
plt.show()




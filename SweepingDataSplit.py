import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir, Ridge
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

DataSplit = np.linspace(0.1, 0.9, 10)
n_runs = 20  # Number of repetitions for each split

mse_means = []
mse_stds = []

# Define Echo State Network
n_reservoir = 100  # Number of reservoir neurons
reservoir1 = Reservoir(n_reservoir, name="res1-1")
readout1 = Ridge(ridge=1e-5, name="readout1-1")
model = reservoir1 >> readout1

for i in DataSplit:
    mse_vals = []
    for _ in range(n_runs):
        # Split dataset into training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=i)

        # Train the ESN on training data
        model.fit(X_train, y_train)

        # Validate the model
        y_pred = model.run(X_val)

        # Compute validation error
        mse = np.mean((y_pred - y_val) ** 2)
        mse_vals.append(mse)

    # Compute mean and standard deviation of MSE
    mse_means.append(np.mean(mse_vals))
    mse_stds.append(np.std(mse_vals))

# Plot results
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1,
    "figure.figsize": (3.5, 2.5),
    "text.usetex": False,
})

fig, ax = plt.subplots()
ax.errorbar(DataSplit, mse_means, yerr=mse_stds, fmt='-o', capsize=5, label="Mean MSE")
ax.set_xlabel('Data Split')
ax.set_ylabel('MSE')
ax.legend()
fig.tight_layout()

plt.savefig("Figs/DataSplit.png", dpi=300, bbox_inches='tight')
plt.show()

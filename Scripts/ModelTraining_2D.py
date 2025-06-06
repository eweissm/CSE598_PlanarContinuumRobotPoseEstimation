"""
Author: Eric Weissman
Date: 4/12

Description:
------------
This script trains and evaluates a Deep Echo State Network (ESN) to predict the curvature of a 2D soft robotic spine
from sensor measurements. The ground truth curvature is represented by polynomial coefficients extracted from video frames.

The script performs the following steps:
1. Loads preprocessed data containing sensor readings, actuator inputs, and corresponding polynomial coefficients.
2. Prepares input features (sensor values) and output labels (polynomial coefficients).
3. Splits the dataset into training and validation sets (90/10 split).
4. Constructs and trains a Deep ESN using the ReservoirPy library.
5. Evaluates the model's performance using mean squared error (MSE) on the validation set.
6. Saves the trained model for future use.
7. Plots example polynomial curves comparing predicted and true curvature profiles.

Key Features:
-------------
- Uses `ReservoirPy` for creating and training the Echo State Network.
- Employs Ridge regression as a readout layer to map reservoir states to target outputs.
- Produces an IEEE-style plot of predicted vs. true curvature using polynomial curves.

Inputs:
-------
- CSV file `ExtractedData/ExtractedData_SmoothSine.csv` containing:
    - Columns 2–7: Sensor readings
    - Columns 8–13: Polynomial coefficients (ground truth)
    - Columns 14–15: Actuator inputs (not used in this script version)

Outputs:
--------
- Trained ESN model saved to `TrainedModels/trained_esn_model.pkl`
- Visualization of a predicted vs. true spine curvature saved as `Figs/fig6.png`
- Printed validation Mean Squared Error (MSE)

Notes:
------
- Only the sensor values are used as input features (actuator data excluded in this version).
- One curve is visualized for comparison, but `NumGraphs` can be adjusted.
- The script applies IEEE-compliant matplotlib formatting for publication-ready figures.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir, Ridge, Input, Output
import matplotlib.pyplot as plt
# import matplotlib
import scienceplots

# Load dataset
file_path = "../ExtractedData/ExtractedData_SmoothSine.csv"  # Replace with actual file path
data = pd.read_csv(file_path)

# Extract features and labels
sensor_data = data.iloc[:, 1:7].values  # Columns 2-7 (zero-indexed as 1:7)
coefficients = data.iloc[:, 7:13].values  # Columns 8-13
actuator_inputs = data.iloc[:, 13:15].values  # Columns 14-15

# Concatenate sensor data and actuator inputs as input features
# X = np.hstack((sensor_data, actuator_inputs))
# y = coefficients  # Ground truth coefficients

# Concatenate Only sensor data as input features
X = sensor_data
y = coefficients  # Ground truth coefficients

# Split dataset into training (90%) and validation (10%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

###################################################################################
# Deep ESN
###################################################################################
#Define Echo State Network
n_reservoir = 200  # Number of reservoir neurons
reservoir1 = Reservoir(n_reservoir, name="res1-1")
readout1 = Ridge(ridge=1e-5, name="readout1-1")
model = reservoir1 >> readout1

# Train the ESN on training data
model.fit(X_train, y_train)

# Validate the model
y_pred = model.run(X_val)

# Compute validation error
mse = np.mean((y_pred - y_val) ** 2)
print(f"Validation MSE: {mse:.6f}")

# Save the trained model
import joblib
joblib.dump(model, "../TrainedModels/trained_esn_model.pkl")



#######################################################
# graph 5 random curved
#######################################################

# Select 5 random samples from the validation set
NumGraphs = 1
# np.random.seed(42)
random_indices = np.random.choice(len(X_val), NumGraphs, replace=False)
X_sample = X_val[random_indices]
y_sample_true = y_val[random_indices]

# Get model predictions
y_sample_pred = model.run(X_sample)

for i in range(NumGraphs):

    poly_func_true = np.poly1d(y_sample_true[i])  # Create a polynomial function
    poly_func_Pred = np.poly1d(y_sample_pred[i])

    # Generate smooth curve points

    x_smooth = np.linspace(100, 775, 100)
    y_smooth_true = poly_func_true(x_smooth)
    y_smooth_Pred = poly_func_Pred(x_smooth)

    # matplotlib.rc('font', size=24)  # Sets the global font size to 20
    # plt.rcParams["text.usetex"] = False
    # plt.rcParams["font.family"] = "sans-serif"
    # plt.rcParams["font.sans-serif"] = ["Arial"]  # Change to a commonly available font
    #
    # plt.style.use(['science', 'ieee'])
    plt.rcParams.update({
        "font.size": 8,  # Adjust font size to IEEE standard
        "axes.labelsize": 8,  # Label size
        "axes.titlesize": 9,  # Title size
        "xtick.labelsize": 7,  # X-axis tick size
        "ytick.labelsize": 7,  # Y-axis tick size
        "legend.fontsize": 7,  # Legend font size
        "lines.linewidth": 1,  # Line width
        "figure.figsize": (3.5, 2.5),  # IEEE single-column figure size
        "text.usetex": False,  # Ensure LaTeX is disabled
    })

    plt.figure(figsize=(2, 4))
    plt.plot(np.flipud(y_smooth_true), x_smooth,'k')
    plt.plot( np.flipud(y_smooth_Pred),x_smooth,'k--')
    plt.axis('equal')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig("Figs/fig6.png", dpi=300, bbox_inches='tight')
plt.show()

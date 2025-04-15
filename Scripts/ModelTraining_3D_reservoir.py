import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat
import joblib
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import time

# Constants
NUM_CENTERS = 3
NUM_MARKERS = 3
POLY_DEGREE = 2
N_RESERVOIR = 2000
RIDGE_ALPHA = 1e-7
NUM_TRIALS = 10  # Number of times to train and average
MODEL_PATH = "../TrainedModels/trained_esn_model_3d.pkl"
CSV_PATH = "../ExtractedData/ManualManipulation_3D_edited.csv"  # Replace with actual file path

# Define marker names and column indices
CENTER_X_COLUMNS = [6, 22, 38, 9, 12, 15, 28, 25, 31, 47, 44, 41]  # X column of each center point
MARKER_NAMES = [f"Center{i}" for i in range(NUM_CENTERS)] + \
               [f"M{i}_{j}" for i in range(NUM_CENTERS) for j in range(NUM_MARKERS)]


def plot_scatter_fit_vs_prediction(y_val, y_pred, frame_index=None):
    """
    Visualize a 3D point ground truth vs prediction for one frame,
    with projections onto XY, YZ, and XZ planes and origin shifted to the first ground truth point.
    """
    if frame_index is None:
        frame_index = np.random.randint(len(y_val))

    # Extract and reshape points
    center_points = np.array([y_val[frame_index, i*3:i*3+3] for i in range(3)])
    prediction = np.array([y_pred[frame_index, i*3:i*3+3] for i in range(3)])

    # Shift everything so the first ground truth point is at the origin
    origin = center_points[0]
    center_points_shifted = center_points - origin
    prediction_shifted = prediction - origin

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    ax.scatter(center_points_shifted[:, 0], center_points_shifted[:, 1], center_points_shifted[:, 2], color='black', label='Center Points', s=50)
    ax.scatter(prediction_shifted[:, 0], prediction_shifted[:, 1], prediction_shifted[:, 2], color='red', label='Prediction', s=50)

    try:
        # Fit splines in shifted space
        tck_gt, _ = splprep(center_points_shifted.T, s=0, k=2)
        spline_gt = splev(np.linspace(0, 1, 100), tck_gt)

        tck_pred, _ = splprep(prediction_shifted.T, s=0, k=2)
        spline_pred = splev(np.linspace(0, 1, 100), tck_pred)

        # Plot splines
        ax.plot(*spline_gt, color='green', linestyle='--', label='Ground Truth', linewidth=2)
        ax.plot(*spline_pred, color='blue', linestyle='--', label='Prediction Fit', linewidth=2)

        # Projections with slight offset
        offset = 0.01

        # XY projection
        ax.plot(spline_gt[0], spline_gt[1], np.full_like(spline_gt[0], np.min(spline_gt[2]) - offset),
                color='green', alpha=0.3, label='GT XY Projection')
        ax.plot(spline_pred[0], spline_pred[1], np.full_like(spline_pred[0], np.min(spline_pred[2]) - offset),
                color='blue', alpha=0.3, label='Pred XY Projection')

        # YZ projection
        ax.plot(np.full_like(spline_gt[1], np.min(spline_gt[0]) - offset), spline_gt[1], spline_gt[2],
                color='green', alpha=0.3, label='GT YZ Projection')
        ax.plot(np.full_like(spline_pred[1], np.min(spline_pred[0]) - offset), spline_pred[1], spline_pred[2],
                color='blue', alpha=0.3, label='Pred YZ Projection')

        # XZ projection
        ax.plot(spline_gt[0], np.full_like(spline_gt[1], np.min(spline_gt[1]) - offset), spline_gt[2],
                color='green', alpha=0.3, label='GT XZ Projection')
        ax.plot(spline_pred[0], np.full_like(spline_pred[1], np.min(spline_pred[1]) - offset), spline_pred[2],
                color='blue', alpha=0.3, label='Pred XZ Projection')

    except Exception as e:
        print("Spline fitting failed:", e)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def remove_nan_frames_from_features_and_targets(X, y):
    """
    Removes any rows where either X or y has NaNs.
    Returns filtered (X, y).
    """
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    return X[valid_mask], y[valid_mask]

def load_marker_data(file_path, marker_names, column_indices):
    """
    Loads and organizes marker data from a CSV.
    """
    data = pd.read_csv(file_path, header=[1, 2, 3, 4, 5, 6, 7], dtype='float32', low_memory=False)
    marker_dict = {
        name: data.iloc[:, column_indices[i]:column_indices[i] + 3].values
        for i, name in enumerate(marker_names)
    }
    return marker_dict, data

def compute_distances(marker_dict, marker1, marker2):
    """
    Computes Euclidean distances between corresponding points of two markers.
    """
    coords1 = marker_dict[marker1]
    coords2 = marker_dict[marker2]
    return np.linalg.norm(coords1 - coords2, axis=1)

def compute_sensor_values(marker_dict, num_centers, num_markers):
    """
    Computes pairwise distances between markers on adjacent centers.
    """
    sensor_values = np.zeros((len(next(iter(marker_dict.values()))), num_markers * (num_centers - 1)))
    for i in range(num_centers - 1):
        for j in range(num_markers):
            idx = i * num_markers + j
            marker1 = f"M{i}_{j}"
            marker2 = f"M{i+1}_{j}"
            sensor_values[:, idx] = compute_distances(marker_dict, marker1, marker2)

    return sensor_values

def fit_3d_polynomial_centers(marker_dict, degree=2):
    """
    Fits a 3D polynomial to the center markers for each frame.
    Returns coefficients stacked as [x_coeffs, y_coeffs, z_coeffs] per frame.
    """
    num_frames = marker_dict["Center0"].shape[0]
    t = np.arange(3)  # parameter values for the centers (0,1,2)

    coeffs = np.zeros((num_frames, (degree + 1) * 3))

    for frame in range(num_frames):
        points = np.array([marker_dict[f"Center{i}"][frame] for i in range(3)])  # shape: (3, 3)
        coeffs_x = np.polyfit(t, points[:, 0], degree)
        coeffs_y = np.polyfit(t, points[:, 1], degree)
        coeffs_z = np.polyfit(t, points[:, 2], degree)
        coeffs[frame] = np.hstack([coeffs_x, coeffs_y, coeffs_z])

    return coeffs

def train_esn(X_train, y_train, n_reservoir, ridge_alpha, seed=None):
    reservoir1 = Reservoir(n_reservoir, seed=seed, name=f"reservoir1_{seed}",lr=0.99, sr=.1)
    reservoir2 = Reservoir(n_reservoir, seed=seed, name=f"reservoir2_{seed}", lr=0.99, sr=.1)

    readout1 = Ridge(ridge=ridge_alpha, name=f"readout1_{seed}")

    # model = reservoir1 >> readout
    concatenate1 = Concat()

    model =[ Input(),  Input() >> reservoir1] >> concatenate1 >> reservoir2>> readout1

    model.fit(X_train, y_train)
    return model

def run_trial(seed=42, plot_one=False):
    marker_dict, _ = load_marker_data(CSV_PATH, MARKER_NAMES, CENTER_X_COLUMNS)

    X_raw = compute_sensor_values(marker_dict, NUM_CENTERS, NUM_MARKERS)
    y_raw = np.hstack([marker_dict[f"Center{i}"] for i in range(NUM_CENTERS)])
    X, y = remove_nan_frames_from_features_and_targets(X_raw, y_raw)

    # Normalize data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=seed)

    model = train_esn(X_train, y_train, n_reservoir=N_RESERVOIR,ridge_alpha =RIDGE_ALPHA, seed=seed)
    y_pred_scaled = model.run(X_val)

    # Inverse transform for evaluation
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_val_orig = y_scaler.inverse_transform(y_val)

    mse = np.mean((y_pred - y_val_orig) ** 2)

    if plot_one:
        plot_scatter_fit_vs_prediction(y_val_orig, y_pred)

    return mse


def main():
    mse_list = []
    time_list = []
    for i in range(NUM_TRIALS):
        tic = time.time()
        seed = 42 + i
        mse = run_trial(seed=seed, plot_one=(1))  # Plot only the first run
        toc = time.time()
        print(f"[Trial {i + 1}] MSE: {mse:.6f}")
        mse_list.append(mse)
        time_list.append(toc - tic)

    time_mean = np.mean(time_list)
    time_std = np.std(time_list)

    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    print(f"\nAverage MSE over {NUM_TRIALS} trials: {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"\nAverage Time over {NUM_TRIALS} trials: {time_mean:.6f} ± {time_std:.6f}")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def calc_mean_erp(trial_points, ecog_data):
    """
    Calculate mean event-related potential (ERP) for finger movements

    Parameters:
    trial_points (str): Path to CSV file with movement start, peak, and finger number
    ecog_data (str): Path to CSV file with time series electrode signal

    Returns:
    numpy.ndarray: 5x1201 matrix of averaged brain signals for each finger
    """
    # Read CSV files
    trial_df = pd.read_csv(
        os.path.join(trial_points),
        dtype={"start": int, "peak": int, "finger": int},
        header=None,
        names=["start", "peak", "finger"],
    )
    ecog_df = pd.read_csv(os.path.join(ecog_data), header=None)
    # Convert to numpy array for easier indexing
    ecog_data = ecog_df.values.flatten()

    # Initialize matrix to store finger ERPs
    fingers_erp_mean = np.zeros((5, 1201))

    # Iterate through fingers 1-5
    for finger in range(1, 6):
        # Find indices of current finger movements
        finger_trials = trial_df[trial_df["finger"] == finger]

        # Store ERP trials for this finger
        finger_erp_trials = []

        # Extract 1201 data points for each movement
        for _, row in finger_trials.iterrows():
            start = row["start"]

            # Check boundary conditions
            if start - 200 >= 0 and start + 1001 <= len(ecog_data):
                # Extract 200 ms before, 1 ms at, and 1000 ms after start
                erp_block = ecog_data[start - 200 : start + 1001]

                # Ensure block is exactly 1201 points
                if len(erp_block) == 1201:
                    finger_erp_trials.append(erp_block)

        # Average across trials for this finger
        if finger_erp_trials:
            fingers_erp_mean[finger - 1] = np.mean(finger_erp_trials, axis=0)

    # Plot averaged brain response for each finger
    plt.figure(figsize=(12, 8))
    time_axis = np.linspace(-200, 1000, 1201)

    for finger in range(5):
        plt.plot(time_axis, fingers_erp_mean[finger], label=f"Finger {finger+1}")

    plt.title("Average Event-Related Potential by Finger")
    plt.xlabel("Time (ms)")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    return fingers_erp_mean


# Test function
calc_mean_erp(
    "mini_project_2_data/events_file_ordered.csv",
    "mini_project_2_data/brain_data_channel_one.csv",
)

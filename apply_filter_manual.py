import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys

# --- Parameters ---
INPUT_FILE = 'resampled_data.txt'
OUTPUT_FILE = 'filtered_data_manual_dot.txt'
OUTPUT_PLOT_FILE = 'filtered_comparison_manual_dot.png'

FS = 50.0  # Sampling frequency (Hz)
CUTOFF_FREQ = 10.0  # Cutoff frequency (Hz)
FILTER_ORDER = 40  # Filter order
NUMTAPS = FILTER_ORDER + 1
# ------------------

# --- Manual Convolution Function ---
def manual_convolution(data, coeffs):
    """Applies FIR filter using manual convolution with numpy dot product."""
    n_data = len(data)
    n_coeffs = len(coeffs)
    output = np.zeros(n_data, dtype=np.float64) # Use float64 for output
    # Reverse coefficients once for dot product usage
    coeffs_rev = coeffs[::-1]
    
    print("Performing manual convolution using np.dot...")
    # Pad the beginning of the data with zeros for handling initial samples
    padded_data = np.pad(data, (n_coeffs - 1, 0), 'constant')

    for n in range(n_data):
        # Extract the relevant slice of padded data
        # The slice corresponds to data[n-k] for k=0..n_coeffs-1
        # which in the padded array is at indices corresponding to original n, n-1, ..., n-M+1
        data_slice = padded_data[n : n + n_coeffs]
        # Perform dot product with reversed coefficients
        output[n] = np.dot(data_slice, coeffs_rev)
        
        # Optional: Print progress
        # if n % (n_data // 10) == 0:
        #     print(f"  ... {int(100*n/n_data)}%")

    print("Manual convolution (np.dot) finished.")
    return output
# ---------------------------------

# --- 1. Load Data ---
print(f"Loading resampled data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE, sep='\t', header=0)
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# --- 2. Design FIR Filter (using scipy.signal for coefficients) ---
print(f"Designing FIR Low-pass filter: fs={FS}Hz, cutoff={CUTOFF_FREQ}Hz, numtaps={NUMTAPS}")
nyquist = FS / 2.0
normalized_cutoff = CUTOFF_FREQ / nyquist
filter_coeffs = signal.firwin(NUMTAPS, normalized_cutoff, window='hamming', pass_zero='lowpass')
print(f"Filter coefficients (first 5): {filter_coeffs[:5]}")

# --- 3. Apply Filter Manually ---
print("Applying filter manually...")
x_data = df['xaccel'].values
y_data = df['yaccel'].values
filtered_x_manual = manual_convolution(x_data, filter_coeffs)
filtered_y_manual = manual_convolution(y_data, filter_coeffs)

# Update column names for clarity
df['xaccel_filtered_manual_dot'] = filtered_x_manual
df['yaccel_filtered_manual_dot'] = filtered_y_manual

# --- 4. Save Filtered Data ---
print(f"Saving manually (np.dot) filtered data to {OUTPUT_FILE}...")
try:
    # Update columns to save
    df_to_save = df[['time', 'xaccel', 'yaccel', 'xaccel_filtered_manual_dot', 'yaccel_filtered_manual_dot']]
    df_to_save.to_csv(OUTPUT_FILE, sep='\t', index=False, float_format='%.9f', header=True)
    print(f"Filtered data saved successfully.")
except Exception as e:
    print(f"Error saving filtered data: {e}")

# --- 5. Visualize Comparison ---
print("Generating comparison plot...")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
# Update title
fig.suptitle('Original (Resampled) vs. Manual (np.dot) Filtered Data', fontsize=16)

# Plot xaccel comparison
axes[0].plot(df['time'], df['xaccel'], label='Original xaccel', color='lightblue', alpha=0.7)
# Update label and column name
axes[0].plot(df['time'], df['xaccel_filtered_manual_dot'], label='Manual (np.dot) Filter xaccel', color='blue')
axes[0].set_ylabel('X Acceleration')
axes[0].set_title('X Acceleration: Original vs. Manual Filter')
axes[0].grid(True)
axes[0].legend()

# Plot yaccel comparison
axes[1].plot(df['time'], df['yaccel'], label='Original yaccel', color='lightcoral', alpha=0.7)
# Update label and column name
axes[1].plot(df['time'], df['yaccel_filtered_manual_dot'], label='Manual (np.dot) Filter yaccel', color='red')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Y Acceleration')
axes[1].set_title('Y Acceleration: Original vs. Manual Filter')
axes[1].grid(True)
axes[1].legend()

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot
try:
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"Comparison plot saved to {OUTPUT_PLOT_FILE}")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show()

print("Process finished.") 
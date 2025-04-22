import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys

# --- Parameters ---
INPUT_FILE = 'resampled_data.txt'
OUTPUT_FILE = 'filtered_data.txt'
OUTPUT_PLOT_FILE = 'filtered_comparison.png'

FS = 50.0  # Sampling frequency (Hz) - should match resampling
CUTOFF_FREQ = 10.0  # Cutoff frequency (Hz)
FILTER_ORDER = 40  # Filter order (Numtaps = Order + 1)
NUMTAPS = FILTER_ORDER + 1
# ------------------

# --- 1. Load Data ---
print(f"Loading resampled data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE, sep='\t', header=0)
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    print("Please run the resampling script first.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# --- 2. Design FIR Filter ---
print(f"Designing FIR Low-pass filter: fs={FS}Hz, cutoff={CUTOFF_FREQ}Hz, numtaps={NUMTAPS}")
# Normalize cutoff frequency to Nyquist frequency (fs/2)
nyquist = FS / 2.0
normalized_cutoff = CUTOFF_FREQ / nyquist

# Design the filter using firwin with a Hamming window
filter_coeffs = signal.firwin(NUMTAPS, normalized_cutoff, window='hamming', pass_zero='lowpass')

# --- 3. Apply Filter ---
print("Applying filter to xaccel and yaccel...")
# Use filtfilt for zero-phase filtering
filtered_x = signal.filtfilt(filter_coeffs, 1.0, df['xaccel'], padlen=FILTER_ORDER)
filtered_y = signal.filtfilt(filter_coeffs, 1.0, df['yaccel'], padlen=FILTER_ORDER)

# Add filtered data to DataFrame
df['xaccel_filtered'] = filtered_x
df['yaccel_filtered'] = filtered_y

print("Filtering complete.")

# --- 4. Save Filtered Data ---
print(f"Saving filtered data to {OUTPUT_FILE}...")
try:
    df.to_csv(OUTPUT_FILE, sep='\t', index=False, float_format='%.9f', header=True)
    print(f"Filtered data saved successfully.")
except Exception as e:
    print(f"Error saving filtered data: {e}")

# --- 5. Visualize Comparison ---
print("Generating comparison plot...")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
fig.suptitle('Original (Resampled) vs. Filtered Data', fontsize=16)

# Plot xaccel comparison
axes[0].plot(df['time'], df['xaccel'], label='Original xaccel', color='lightblue', alpha=0.7)
axes[0].plot(df['time'], df['xaccel_filtered'], label='Filtered xaccel', color='blue')
axes[0].set_ylabel('X Acceleration')
axes[0].set_title('X Acceleration: Original vs. Filtered')
axes[0].grid(True)
axes[0].legend()

# Plot yaccel comparison
axes[1].plot(df['time'], df['yaccel'], label='Original yaccel', color='lightcoral', alpha=0.7)
axes[1].plot(df['time'], df['yaccel_filtered'], label='Filtered yaccel', color='red')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Y Acceleration')
axes[1].set_title('Y Acceleration: Original vs. Filtered')
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
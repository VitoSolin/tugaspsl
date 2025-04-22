import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Parameters ---
INPUT_FILE = 'resampled_data.txt'
OUTPUT_PLOT_FILE = 'frequency_spectrum.png'
FS = 50.0  # Sampling frequency (Hz)
# ------------------

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

# Extract signals
x_signal = df['xaccel'].values
y_signal = df['yaccel'].values
N = len(x_signal) # Number of samples

if N == 0:
    print("Error: No data found in the file.")
    sys.exit(1)

print(f"Analyzing {N} samples with Fs = {FS} Hz.")

# --- 2. Perform FFT ---
print("Performing FFT...")
# Calculate FFT
fft_x = np.fft.fft(x_signal)
fft_y = np.fft.fft(y_signal)

# Calculate corresponding frequencies
# np.fft.fftfreq(N, d=1/FS) gives frequencies for the whole spectrum (positive and negative)
freqs = np.fft.fftfreq(N, d=1/FS)

# --- 3. Calculate Magnitude and Find Dominant Frequency ---
# We only need the positive frequency part for magnitude analysis
positive_freq_indices = np.where(freqs >= 0)[0]
freqs_pos = freqs[positive_freq_indices]

# Calculate magnitude (absolute value of complex FFT output)
# Normalize by N/2 for amplitude (optional, doesn't affect dominant freq finding)
# We take only the positive frequency part
magnitude_x = np.abs(fft_x[positive_freq_indices])
magnitude_y = np.abs(fft_y[positive_freq_indices])

# Find the index of the maximum magnitude, *excluding* the DC component (index 0)
# Handle cases with very few points where excluding index 0 might be problematic
if len(freqs_pos) > 1:
    dominant_idx_x = np.argmax(magnitude_x[1:]) + 1 # Add 1 because we skipped index 0
    dominant_freq_x = freqs_pos[dominant_idx_x]
    dominant_mag_x = magnitude_x[dominant_idx_x]

    dominant_idx_y = np.argmax(magnitude_y[1:]) + 1 # Add 1 because we skipped index 0
    dominant_freq_y = freqs_pos[dominant_idx_y]
    dominant_mag_y = magnitude_y[dominant_idx_y]
else: # Only DC component exists or just one point
    dominant_idx_x = 0
    dominant_freq_x = freqs_pos[0]
    dominant_mag_x = magnitude_x[0]
    dominant_idx_y = 0
    dominant_freq_y = freqs_pos[0]
    dominant_mag_y = magnitude_y[0]

print("\n--- Dominant Frequencies (excluding DC) ---")
print(f"X Accel: {dominant_freq_x:.2f} Hz (Magnitude: {dominant_mag_x:.2f})")
print(f"Y Accel: {dominant_freq_y:.2f} Hz (Magnitude: {dominant_mag_y:.2f})")

# --- 4. Visualize Spectrum ---
print("\nGenerating frequency spectrum plot...")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
fig.suptitle('Frequency Spectrum (Resampled Data)', fontsize=16)

# Plot X Spectrum (positive frequencies only, excluding DC for visual clarity)
if len(freqs_pos) > 1:
    axes[0].plot(freqs_pos[1:], magnitude_x[1:], label='xaccel', color='blue')
    # Highlight dominant frequency
    axes[0].axvline(dominant_freq_x, color='blue', linestyle='--', alpha=0.7, label=f'Dom X: {dominant_freq_x:.2f} Hz')
else:
    axes[0].plot(freqs_pos, magnitude_x, label='xaccel', color='blue') # Plot DC if it's the only point
axes[0].set_ylabel('Magnitude')
axes[0].set_title('X Acceleration Spectrum')
axes[0].grid(True)
axes[0].legend()

# Plot Y Spectrum (positive frequencies only, excluding DC)
if len(freqs_pos) > 1:
    axes[1].plot(freqs_pos[1:], magnitude_y[1:], label='yaccel', color='red')
    # Highlight dominant frequency
    axes[1].axvline(dominant_freq_y, color='red', linestyle='--', alpha=0.7, label=f'Dom Y: {dominant_freq_y:.2f} Hz')
else:
    axes[1].plot(freqs_pos, magnitude_y, label='yaccel', color='red')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Y Acceleration Spectrum')
axes[1].grid(True)
axes[1].legend()

# Adjust layout and x-axis limit (up to Nyquist)
plt.xlim(0, FS / 2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot
try:
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"Spectrum plot saved to {OUTPUT_PLOT_FILE}")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show()

print("Frequency analysis finished.")
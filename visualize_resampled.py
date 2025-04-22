import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- Parameters ---
INPUT_FILE = 'resampled_data.txt'
OUTPUT_PLOT_FILE = 'resampled_visualization.png'
# ------------------

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

print(f"Generating plot...")

# Create figure and axes for subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
fig.suptitle('Resampled Accelerometer Data (50 Hz)', fontsize=16)

# Plot xaccel
axes[0].plot(df['time'], df['xaccel'], label='xaccel', color='blue')
axes[0].set_ylabel('X Acceleration')
axes[0].set_title('X Acceleration vs. Time')
axes[0].grid(True)
axes[0].legend()

# Plot yaccel
axes[1].plot(df['time'], df['yaccel'], label='yaccel', color='red')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Y Acceleration')
axes[1].set_title('Y Acceleration vs. Time')
axes[1].grid(True)
axes[1].legend()

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle

# Save the plot
try:
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"Plot saved to {OUTPUT_PLOT_FILE}")
except Exception as e:
    print(f"Error saving plot: {e}")

# Optional: Display the plot if running in an environment that supports it
# plt.show() 
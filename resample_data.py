import pandas as pd
import numpy as np

# --- Parameters ---
INPUT_FILE = 'datates.txt'
OUTPUT_FILE = 'resampled_data.txt'
TARGET_FS = 50.0  # Target sampling frequency in Hz
# ------------------

print(f"Loading data from {INPUT_FILE}...")
try:
    # Read data, keeping default index
    df = pd.read_csv(INPUT_FILE, sep='\t', header=0)
    print("File read successfully. First 5 rows:")
    print(df.head().to_string())
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

if 'time' not in df.columns:
    print(f"Error: Column 'time' not found in {INPUT_FILE}.")
    exit()

# Calculate cumulative time (as float seconds)
df['cumulative_time'] = df['time'].cumsum()
original_time_sec = df['cumulative_time'].values
original_xaccel = df['xaccel'].values
original_yaccel = df['yaccel'].values

print("\n--- DEBUG: Original Time and Accel (first 10) ---")
print("Time (s):", original_time_sec[:10])
print("X Accel:", original_xaccel[:10])
print("--- END DEBUG ---")

# Define target time points in seconds
target_period_sec = 1.0 / TARGET_FS
start_time_sec = original_time_sec[0]
end_time_sec = original_time_sec[-1]
target_time_sec = np.arange(start_time_sec, end_time_sec, target_period_sec)

print(f"\nResampling data to {TARGET_FS} Hz using np.interp...")

# Interpolate using numpy
resampled_xaccel = np.interp(target_time_sec, original_time_sec, original_xaccel)
resampled_yaccel = np.interp(target_time_sec, original_time_sec, original_yaccel)

print("\n--- DEBUG: Resampled values (first 10) ---")
print("Time (s):", target_time_sec[:10])
print("X Accel:", resampled_xaccel[:10])
print("Y Accel:", resampled_yaccel[:10])
print("--- END DEBUG ---")

# Create the final DataFrame
resampled_df = pd.DataFrame({
    'time': target_time_sec,
    'xaccel': resampled_xaccel,
    'yaccel': resampled_yaccel
})

print(f"\nSaving resampled data to {OUTPUT_FILE}...")
try:
    resampled_df.to_csv(OUTPUT_FILE, sep='\t', index=False, float_format='%.9f', header=True)
    print("Save complete.")
except Exception as e:
    print(f"Error saving file: {e}")
    exit()

print("\n--- Verification Step ---")
try:
    print(f"Reading back {OUTPUT_FILE} for verification...")
    df_verify = pd.read_csv(OUTPUT_FILE, sep='\t', header=0)
    print("File read back successfully.")
    print("\n--- DEBUG: Data AFTER reading back (first 10 rows) ---")
    print(df_verify.head(10).to_string())
    print("--- END DEBUG ---")
except Exception as e:
    print(f"Error reading back file for verification: {e}")
# ------------------------

print(f"\nResampling and verification complete. Resampled data has {len(resampled_df)} points.")

# print("\nFirst 5 rows of resampled data:")
# print(resampled_df.head().to_string()) # Commented out, replaced by verification step 
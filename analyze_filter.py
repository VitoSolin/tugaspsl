import numpy as np
print("Imported numpy")
from scipy import signal
print("Imported signal")
import matplotlib.pyplot as plt
print("Imported pyplot")

print("Starting script analyze_filter.py...")

# --- Parameters (same as filter design) ---
print("Defining parameters...")
FS = 50.0
CUTOFF_FREQ = 10.0  # Added back
FILTER_ORDER = 40   # Added back
NUMTAPS = FILTER_ORDER + 1 # Added back
WINDOW_TYPE = 'hamming' # Added back
# ------------------

# --- 1. Design Filter Coefficients ---
# print("Designing FIR Low-pass filter: ...") # Moved after
nyquist = FS / 2.0
normalized_cutoff = CUTOFF_FREQ / nyquist
print(f"Calculated nyquist={nyquist}, normalized_cutoff={normalized_cutoff}") # Added print
try:
    print("Attempting signal.firwin...") # Added print
    filter_coeffs = signal.firwin(NUMTAPS, normalized_cutoff, window=WINDOW_TYPE, pass_zero='lowpass')
    print(f"Designing FIR Low-pass filter: fs={FS}Hz, cutoff={CUTOFF_FREQ}Hz, numtaps={NUMTAPS}, window={WINDOW_TYPE}") # Moved here
    print(f"Filter coefficients generated successfully (first 5): {filter_coeffs[:5]}") # Added print
except Exception as e:
    print(f"Error during signal.firwin: {e}") # Added error handling
    exit()

# --- 2. Calculate Frequency Response ---
print("Calculating frequency response...")
try:
    print("Attempting signal.freqz...")
    w, h = signal.freqz(filter_coeffs, worN=8000) 
    print(f"signal.freqz finished. w shape: {w.shape}, h shape: {h.shape}")
except Exception as e:
    print(f"Error during signal.freqz: {e}")
    exit()

# --- 3. Analyze Response --- 
print("Analyzing response characteristics...") # Re-enabled

# Define approximate passband and stopband frequency ranges for analysis
passband_freq_limit = CUTOFF_FREQ * 0.8 # e.g., up to 8 Hz
stopband_freq_start = CUTOFF_FREQ * 1.2 # e.g., from 12 Hz

# Recalculate these here for analysis
frequencies_hz = (w / np.pi) * nyquist
magnitude_db = 20 * np.log10(np.abs(h))

# Find indices corresponding to these frequency ranges
passband_indices = np.where(frequencies_hz <= passband_freq_limit)[0]
stopband_indices = np.where(frequencies_hz >= stopband_freq_start)[0]

# Default values in case analysis fails
passband_ripple_db = np.nan
min_attenuation_db = np.nan
max_passband_db = 0
min_passband_db = 0

# Passband Ripple Analysis
if len(passband_indices) > 0:
    passband_mags_db = magnitude_db[passband_indices]
    # Check for non-finite values before calculating max/min
    if np.all(np.isfinite(passband_mags_db)):
        max_passband_db = np.max(passband_mags_db)
        min_passband_db = np.min(passband_mags_db)
        passband_ripple_db = max_passband_db - min_passband_db
        print(f"- Passband (0-{passband_freq_limit:.1f} Hz):")
        print(f"  - Max Gain: {max_passband_db:.4f} dB")
        print(f"  - Min Gain: {min_passband_db:.4f} dB")
        print(f"  - Approx Passband Ripple: {passband_ripple_db:.4f} dB")
    else:
        print("- Passband analysis skipped due to non-finite values.")
else:
    print("- Could not analyze passband (no indices).")

# Stopband Attenuation Analysis
if len(stopband_indices) > 0:
    stopband_mags_db = magnitude_db[stopband_indices]
    if np.all(np.isfinite(stopband_mags_db)):
        min_attenuation_db = np.max(stopband_mags_db) # Max gain = min attenuation
        print(f"- Stopband ({stopband_freq_start:.1f}-{nyquist:.1f} Hz):")
        print(f"  - Minimum Attenuation (Max Stopband Gain): {min_attenuation_db:.4f} dB")
    else:
        print("- Stopband analysis skipped due to non-finite values.")
else:
    print("- Could not analyze stopband (no indices).")

# --- 4. Plot Frequency Response ---
print("Generating frequency response plot...") # Re-enabled
try:
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_hz, magnitude_db)
    plt.title(f'FIR Filter Frequency Response ({WINDOW_TYPE} Window, {NUMTAPS} taps)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim(-100, 5) # Adjust y-axis limits
    plt.grid(True, which='both', axis='both')
    plt.axvline(CUTOFF_FREQ, color='red', linestyle='--', alpha=0.7, label=f'Cutoff ({CUTOFF_FREQ} Hz)')
    # Re-enable annotation lines using calculated values
    if not np.isnan(min_attenuation_db):
        plt.axhline(min_attenuation_db, color='green', linestyle=':', alpha=0.7, label=f'Min Atten ({min_attenuation_db:.2f} dB)')
    if not np.isnan(passband_ripple_db):
        plt.axhline(max_passband_db, color='orange', linestyle=':', alpha=0.7, label=f'Passband Max/Min')
        plt.axhline(min_passband_db, color='orange', linestyle=':', alpha=0.7)
    plt.legend()

    OUTPUT_PLOT_FILE = 'filter_frequency_response.png' # Define here
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"Frequency response plot saved to {OUTPUT_PLOT_FILE}")
except Exception as e:
    print(f"Error during plotting or saving: {e}") # More specific error message

print("Filter analysis finished.") 
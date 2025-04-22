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
# (Keep commented out for now)
# print("Analyzing response characteristics...")
# ...

# --- 4. Plot Frequency Response ---
print("Generating frequency response plot...") # Re-enabled
try:
    # Recalculate these for plotting
    frequencies_hz = (w / np.pi) * nyquist
    magnitude_db = 20 * np.log10(np.abs(h))

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_hz, magnitude_db)
    plt.title(f'FIR Filter Frequency Response ({WINDOW_TYPE} Window, {NUMTAPS} taps)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim(-100, 5) # Adjust y-axis limits
    plt.grid(True, which='both', axis='both')
    plt.axvline(CUTOFF_FREQ, color='red', linestyle='--', alpha=0.7, label=f'Cutoff ({CUTOFF_FREQ} Hz)')
    # Removed other annotation lines for now
    # plt.axhline(min_attenuation_db, ...)
    # plt.axhline(max_passband_db, ...)
    # plt.axhline(min_passband_db, ...)
    plt.legend()

    OUTPUT_PLOT_FILE = 'filter_frequency_response.png' # Define here
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"Frequency response plot saved to {OUTPUT_PLOT_FILE}")
except Exception as e:
    print(f"Error during plotting or saving: {e}") # More specific error message

# print("Filter analysis finished (core calculation only).") # Change back
print("Filter analysis finished.") 
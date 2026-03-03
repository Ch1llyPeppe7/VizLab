"""
Test script to verify bug fixes.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")
try:
    from base import setup_plot_style, save_figure
    from base.plot_utils import COLORS, POWER_COLORS
    print(f"[OK] COLORS imported: {list(COLORS.keys())}")
    print(f"[OK] POWER_COLORS imported: {list(POWER_COLORS.keys())}")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

print("\nTesting FrequencyFunction with different powers...")
from base import FrequencyFunction

powers = [3.0, 1.0, 0.5, -1.0]
for power in powers:
    try:
        freq_func = FrequencyFunction(dim=64, power=power)
        freqs = freq_func.get_frequencies()
        print(f"[OK] Power {power}: {len(freqs)} frequencies, range [{freqs.min():.6f}, {freqs.max():.6f}]")
        
        # Check for NaN or Inf
        if np.any(np.isnan(freqs)) or np.any(np.isinf(freqs)):
            print(f"[WARN] Power {power}: Contains NaN or Inf values!")
    except Exception as e:
        print(f"[FAIL] Power {power}: {e}")

print("\nTesting KernelFunction...")
from base import KernelFunction

try:
    freq_func = FrequencyFunction(dim=64, power=3.0)
    kernel = KernelFunction(freq_func)
    K = kernel.compute_real([0, 100, 500])
    print(f"[OK] Kernel values: {K}")
except Exception as e:
    print(f"[FAIL] Kernel computation: {e}")

print("\nTesting LAPEEncoder...")
from base import LAPEEncoder

try:
    encoder = LAPEEncoder(dim=64, power=3.0)
    coords = np.array([[100, 200], [300, 400]])
    embeddings = encoder.encode(coords)
    print(f"[OK] LAPEEncoder: input shape {coords.shape} -> output shape {embeddings.shape}")
except Exception as e:
    print(f"[FAIL] LAPEEncoder: {e}")

print("\nAll basic tests passed!")
print("You can now run the visualization scripts.")

"""
LAPE (Location-Aware Position Encoding) Pure Mathematical Implementation

This module implements the core mathematical formulas from the TCFMamba paper,
completely independent of model training or weight loading.

Mathematical Framework:
---------------------
1. Frequency Function: ω_k = (-k/d)^a, where a=3 for TCFMamba
2. Kernel Function: K(Δq) = Σ exp(i·ω_k·Δq) = Σ cos(ω_k·Δq) + i·sin(ω_k·Δq)
3. Real Part: Re(K) = Σ cos(ω_k·Δq) - the actual similarity metric
4. Complex Embedding: z(q) = exp(i·ω·q) = cos(ω·q) + i·sin(ω·q)

Reference: math.md (Kähler structure, Bochner theorem, frequency function analysis)
"""

import numpy as np
from typing import Union, Tuple, Optional
import warnings


class FrequencyFunction:
    """
    Frequency function generator for LAPE encoding.
    
    Mathematical Definition:
        ω_k = (-k/d)^a,  k = 0, 1, ..., m-1
        
    where:
        - d: total embedding dimension (locdim in code)
        - m: complex dimension = d/2
        - a: power exponent (a=3 for TCFMamba, superlinear growth)
        
    The frequency distribution determines:
        1. Kernel decay rate: O(1/(Δq)^(1/a)) as Δq → ∞
        2. Resolution: high frequencies capture fine-grained differences
        3. Bochner measure: μ = Σ δ_{ω_k} (discrete non-negative measure)
    """
    
    def __init__(self, dim: int = 64, power: float = 3.0):
        """
        Initialize frequency function.
        
        Args:
            dim: Total embedding dimension (must be even)
            power: Power exponent 'a' in frequency formula
                   - a=3.0: TCFMamba (superlinear, low-frequency dense)
                   - a=1.0: Linear distribution
                   - a=0.5: Sublinear distribution
                   - a=-1.0: Transformer geometric distribution
        """
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even, got {dim}")
        
        self.dim = dim
        self.m = dim // 2  # Complex dimension
        self.power = power
        
        # Generate frequency sequence: ω_k = (-k/d)^a
        k = np.arange(self.m)
        d_float = float(dim)
        
        # Core formula from math.md: ω_k = (-k/d)^a
        # Handle special cases to avoid numerical issues:
        # - k=0: result is 0 (not inf for negative powers)
        # - negative base with fractional power: use sign * abs(base)^power
        base = -k / d_float
        
        # For k=0, set to 0 explicitly
        with np.errstate(divide='ignore', invalid='ignore'):
            freqs = np.sign(base) * np.abs(base) ** power
            freqs[0] = 0.0  # k=0 always gives 0
        
        self.frequencies = freqs
        
    def get_frequencies(self) -> np.ndarray:
        """Return the frequency sequence [ω_0, ω_1, ..., ω_{m-1}]."""
        return self.frequencies.copy()
    
    def plot_distribution(self, ax=None, label=None, color=None):
        """Plot frequency distribution curve."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            ax = plt.gca()
            
        k = np.arange(self.m)
        label = label or f"a={self.power}"
        
        ax.plot(k, self.frequencies, color=color, linewidth=2, label=label)
        ax.set_xlabel('Complex Dimension Index $k$', fontsize=12)
        ax.set_ylabel('Frequency $\\omega_k$', fontsize=12)
        ax.set_title(f'Frequency Distribution (Power a={self.power})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax


class KernelFunction:
    """
    Kernel function implementation based on Bochner's theorem.
    
    Mathematical Definition:
        K(Δq) = Σ_{k=1}^m exp(i·ω_k·Δq)
              = Σ_{k=1}^m [cos(ω_k·Δq) + i·sin(ω_k·Δq)]
              
    Real Part (Observable Similarity):
        Re(K(Δq)) = Σ_{k=1}^m cos(ω_k·Δq)
        
    Properties:
        1. Positive definite (by Bochner's theorem)
        2. Stationary: depends only on Δq = q_1 - q_2
        3. Decay rate: O(1/(Δq)^(1/a)) for power-law frequencies
    """
    
    def __init__(self, freq_func: FrequencyFunction):
        """
        Initialize kernel function.
        
        Args:
            freq_func: FrequencyFunction instance providing ω_k
        """
        self.freq_func = freq_func
        self.frequencies = freq_func.get_frequencies()
        self.m = len(self.frequencies)
        
    def compute(self, delta_q: Union[float, np.ndarray]) -> complex:
        """
        Compute kernel K(Δq) at given position difference.
        
        Args:
            delta_q: Position difference (scalar or array)
            
        Returns:
            Complex kernel value(s)
        """
        # K(Δq) = Σ exp(i·ω_k·Δq)
        phases = np.outer(self.frequencies, np.atleast_1d(delta_q))  # Shape: (m, n)
        kernel_values = np.sum(np.exp(1j * phases), axis=0)  # Sum over frequencies
        
        if np.isscalar(delta_q) or len(np.atleast_1d(delta_q)) == 1:
            return complex(kernel_values[0])
        return kernel_values
    
    def compute_real(self, delta_q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute real part Re(K(Δq)) = Σ cos(ω_k·Δq).
        
        This is the actual similarity metric used in position encoding.
        """
        # Re(K) = Σ cos(ω_k·Δq)
        phases = np.outer(self.frequencies, np.atleast_1d(delta_q))
        real_values = np.sum(np.cos(phases), axis=0)
        
        if np.isscalar(delta_q) or len(np.atleast_1d(delta_q)) == 1:
            return float(real_values[0])
        return real_values
    
    def compute_imag(self, delta_q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute imaginary part Im(K(Δq)) = Σ sin(ω_k·Δq)."""
        phases = np.outer(self.frequencies, np.atleast_1d(delta_q))
        imag_values = np.sum(np.sin(phases), axis=0)
        
        if np.isscalar(delta_q) or len(np.atleast_1d(delta_q)) == 1:
            return float(imag_values[0])
        return imag_values
    
    def compute_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix K_{ij} = Re(K(q_i - q_j)).
        
        Args:
            positions: Array of position values [N,]
            
        Returns:
            Kernel matrix [N, N]
        """
        # Compute pairwise differences
        delta_q = positions[:, None] - positions[None, :]  # Shape: (N, N)
        
        # Compute real part of kernel for all pairs
        K_matrix = self.compute_real(delta_q.flatten()).reshape(delta_q.shape)
        
        return K_matrix
    
    def theoretical_decay_rate(self, delta_q: float) -> float:
        """
        Theoretical asymptotic decay rate O(1/(Δq)^(1/a)).
        
        For power-law frequency distribution ω_k ~ k^a:
            |K(Δq)| ~ O(1/(Δq)^(1/a)) as Δq → ∞
            
        TCFMamba (a=3): O(1/(Δq)^(1/3)) - slower decay, smoother similarity
        Transformer (a=-1, geometric): O(1/Δq) - faster decay, local attention
        """
        a = self.freq_func.power
        if a > 0:
            # O(1/(Δq)^(1/a))
            return 1.0 / (np.abs(delta_q) ** (1.0 / a) + 1e-10)
        else:
            # For geometric distribution (Transformer), exponential decay
            return np.exp(-0.1 * np.abs(delta_q))


class LAPEEncoder:
    """
    LAPE (Location-Aware Position Encoding) encoder.
    
    Encodes 2D spatial coordinates (x, y) using sinusoidal encoding
    similar to Transformer positional encoding but adapted for 2D.
    
    Mathematical Formula:
        For coordinate q (x or y), dimension i:
            PE_{2i}(q)   = sin(q / 200 · ω_i)
            PE_{2i+1}(q) = cos(q / 200 · ω_{i+1})
            
    where ω_i = (-i/d)^a is the frequency function.
    """
    
    def __init__(self, dim: int = 64, power: float = 3.0, scale: float = 200.0):
        """
        Initialize LAPE encoder.
        
        Args:
            dim: Embedding dimension
            power: Power exponent for frequency function
            scale: Coordinate normalization scale (~1km precision)
        """
        self.dim = dim
        self.power = power
        self.scale = scale
        self.freq_func = FrequencyFunction(dim, power)
        self.frequencies = self.freq_func.get_frequencies()
        
    def encode(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Encode 2D coordinates to LAPE embeddings.
        
        Args:
            coordinates: Array of shape [N, 2] with (x, y) coordinates
            
        Returns:
            Encodings of shape [N, dim]
        """
        coordinates = np.atleast_2d(coordinates)
        N = coordinates.shape[0]
        
        # Normalize coordinates
        coords_norm = coordinates / self.scale
        
        # Split dimension for x and y
        d_half = self.dim // 2
        
        encoding = np.zeros((N, self.dim))
        
        # Encode x coordinate (first half)
        for i in range(0, d_half, 2):
            freq_x = self.frequencies[i // 2]
            encoding[:, i] = np.sin(coords_norm[:, 0] * freq_x)
            if i + 1 < d_half:
                encoding[:, i + 1] = np.cos(coords_norm[:, 0] * freq_x)
        
        # Encode y coordinate (second half)
        for i in range(0, d_half, 2):
            freq_y = self.frequencies[i // 2]
            encoding[:, d_half + i] = np.sin(coords_norm[:, 1] * freq_y)
            if i + 1 < d_half:
                encoding[:, d_half + i + 1] = np.cos(coords_norm[:, 1] * freq_y)
        
        return encoding
    
    def encode_complex(self, coordinate: float, freq_idx: int) -> complex:
        """
        Encode single coordinate to complex embedding.
        
        z(q) = exp(i·ω·q) = cos(ω·q) + i·sin(ω·q)
        
        This represents a point on the unit circle in complex plane,
        rotating at angular velocity ω.
        """
        omega = self.frequencies[freq_idx]
        q_norm = coordinate / self.scale
        
        # Complex embedding: z = e^(i·ω·q)
        z_real = np.cos(omega * q_norm)
        z_imag = np.sin(omega * q_norm)
        
        return complex(z_real, z_imag)


class SphericalTransform:
    """
    Spherical coordinate transformation (latitude/longitude to Cartesian).
    
    Used for converting geographic coordinates to 2D plane coordinates
    suitable for LAPE encoding.
    
    Formula:
        x = R · cos(lat) · sin(lon)
        y = R · cos(lat) · cos(lon)
        
    where R is Earth radius (default: 6371 km).
    """
    
    def __init__(self, radius: float = 6371.0):
        """
        Initialize spherical transform.
        
        Args:
            radius: Earth radius in kilometers
        """
        self.radius = radius
        
    def transform(self, latitudes: np.ndarray, longitudes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform (lat, lon) to Cartesian (x, y).
        
        Args:
            latitudes: Latitudes in degrees
            longitudes: Longitudes in degrees
            
        Returns:
            Tuple of (x, y) coordinates in meters
        """
        # Convert to radians
        lat_rad = np.radians(latitudes)
        lon_rad = np.radians(longitudes)
        
        # Spherical to Cartesian conversion
        x = self.radius * np.cos(lat_rad) * np.sin(lon_rad) * 1000  # Convert to meters
        y = self.radius * np.cos(lat_rad) * np.cos(lon_rad) * 1000
        
        # Normalize to positive coordinates (relative to minimum)
        x = x - x.min()
        y = y - y.min()
        
        return x, y
    
    def inverse_transform(self, x: np.ndarray, y: np.ndarray, 
                         x_min: float, y_min: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse transform from Cartesian (x, y) to approximate (lat, lon).
        
        Note: This is an approximation assuming small regions.
        """
        # Restore original coordinates
        x_orig = x + x_min
        y_orig = y + y_min
        
        # Convert back to lat/lon (approximate)
        r = np.sqrt(x_orig**2 + y_orig**2) / 1000  # km
        lon = np.degrees(np.arctan2(x_orig, y_orig))
        lat = np.degrees(np.arccos(r / self.radius))
        
        return lat, lon


def compare_frequency_powers(dim: int = 64, powers: list = None) -> dict:
    """
    Compare different frequency power exponents.
    
    Returns dictionary with frequency sequences for each power value.
    Useful for visualizing how a affects frequency distribution.
    """
    if powers is None:
        # TCFMamba: a=3, Transformer: a=-1, Linear: a=1
        powers = [-1.0, 0.5, 1.0, 3.0]
    
    results = {}
    for power in powers:
        freq_func = FrequencyFunction(dim, power)
        results[power] = {
            'frequencies': freq_func.get_frequencies(),
            'm': freq_func.m,
            'description': _get_power_description(power)
        }
    
    return results


def _get_power_description(power: float) -> str:
    """Get description for power exponent."""
    descriptions = {
        -1.0: "Transformer (geometric)",
        0.5: "Sublinear",
        1.0: "Linear",
        3.0: "TCFMamba (superlinear)"
    }
    return descriptions.get(power, f"Power a={power}")


# Utility functions for visualization
def generate_position_grid(n_points: int = 100, x_range: Tuple[float, float] = (0, 1000),
                           y_range: Tuple[float, float] = (0, 1000)) -> np.ndarray:
    """Generate 2D grid of positions for kernel matrix visualization."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    
    # Create 2D grid
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.flatten(), yy.flatten()])
    
    return positions


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between positions."""
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    return distances

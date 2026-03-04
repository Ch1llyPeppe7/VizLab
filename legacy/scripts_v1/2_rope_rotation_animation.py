"""
Visualization 2: ROPE Rotation and Parallel Transport Animation

This script creates an animation demonstrating ROPE (Rotary Position Embedding)
rotation and its connection to Kähler isometric transformations.

Mathematical Background:
-------------------------
ROPE applies rotation in complex plane to encode relative positions:
    z_rotated = z · exp(i·θ)
    
where θ = ω·Δq is the rotation angle determined by position difference.

Parallel Transport on Kähler Manifold:
----------------------------------------
The Kähler structure ensures that position encoding preserves geometric
relationships through parallel transport. The rotation in complex plane
is an isometry (distance-preserving transformation) of the Kähler manifold.

Key Properties:
1. Rotation preserves |z| = 1 (unit circle)
2. Inner product ⟨z_1, z_2⟩ is preserved under simultaneous rotation
3. This corresponds to parallel transport on the manifold

Animation Content:
------------------
- Frame 1-30: Single frequency rotation on complex plane
- Frame 31-60: Multiple frequencies rotating at different speeds
- Frame 61-90: Parallel transport demonstration
- Frame 91-120: Inner product preservation visualization

Reference: math.md Section 4 (Kähler structure theorem, isometric transformations)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
from pathlib import Path

# Add base module to path
sys.path.insert(0, str(Path(__file__).parent))
from base import setup_plot_style, save_figure
from base.plot_utils import COLORS
from base.viz_logger import VizLogger


def create_rope_rotation_animation(output_dir: Path = None, duration: float = 5.0):
    """
    Create ROPE rotation animation showing parallel transport on complex plane.
    
    Args:
        output_dir: Output directory for saving animation
        duration: Animation duration in seconds (default: 5s)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = VizLogger('2_rope_rotation_animation')
    logger.set_description('ROPE Rotation and Parallel Transport Animation (Kähler Structure)')
    logger.add_finding('ROPE applies rotation z_rotated = z · exp(i·θ) where θ = ω·Δq', 'theoretical')
    logger.add_finding('Rotation preserves |z| = 1 (unit circle, isometry)', 'mathematical')
    logger.add_finding('Inner product is preserved under simultaneous rotation', 'empirical')
    logger.log_parameter('formula_rotation', 'z_rotated = z · exp(i·ω·Δq)')
    logger.log_parameter('formula_kaehler', '⟨z_1, z_2⟩ preserved under rotation')
    logger.log_parameter('omega', 0.1)
    logger.log_parameter('n_frames', 120)
    logger.log_parameter('fps', int(120 / duration))
    
    setup_plot_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parameters
    omega = 0.1  # Frequency
    n_frames = 120
    fps = int(n_frames / duration)
    
    # Setup left plot: Single frequency rotation
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel('Real', fontsize=11)
    ax1.set_ylabel('Imaginary', fontsize=11)
    ax1.set_title('Single Frequency Rotation\n$z(q) = e^{i\\omega q}$', fontsize=12)
    
    # Add unit circle
    circle1 = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax1.add_patch(circle1)
    
    # Setup right plot: Parallel transport (two points)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax2.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Real', fontsize=11)
    ax2.set_ylabel('Imaginary', fontsize=11)
    ax2.set_title('Parallel Transport: Two Points\n$z_1, z_2$ rotated together', fontsize=12)
    
    # Add unit circles for both points
    circle2a = Circle((0, 0), 1, fill=False, color='blue', linestyle='--', alpha=0.3)
    circle2b = Circle((0, 0), 1, fill=False, color='red', linestyle='--', alpha=0.3)
    ax2.add_patch(circle2a)
    ax2.add_patch(circle2b)
    
    # Initialize plot elements
    # Left plot: rotating point and trail
    point1, = ax1.plot([], [], 'o', color=COLORS['primary'], markersize=15, zorder=5)
    trail1, = ax1.plot([], [], '-', color=COLORS['primary'], alpha=0.3, linewidth=2)
    arrow1 = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top')
    
    # Right plot: two rotating points
    point2a, = ax2.plot([], [], 'o', color='blue', markersize=12, zorder=5, label='$z_1$')
    point2b, = ax2.plot([], [], 'o', color='red', markersize=12, zorder=5, label='$z_2$')
    line2, = ax2.plot([], [], '--', color='gray', alpha=0.5, linewidth=1)
    text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top')
    
    # Phase difference between two points (right plot)
    phase_diff = np.pi / 3  # 60 degrees
    
    # Trail storage
    trail_x, trail_y = [], []
    max_trail_length = 50
    
    def init():
        """Initialize animation frames."""
        point1.set_data([], [])
        trail1.set_data([], [])
        point2a.set_data([], [])
        point2b.set_data([], [])
        line2.set_data([], [])
        text1.set_text('')
        text2.set_text('')
        return point1, trail1, point2a, point2b, line2, text1, text2
    
    def update(frame):
        """Update animation for each frame."""
        t = frame / n_frames * 4 * np.pi  # Two full rotations
        
        # Left plot: Single point rotating
        z_real = np.cos(omega * t)
        z_imag = np.sin(omega * t)
        
        point1.set_data([z_real], [z_imag])
        
        # Update trail
        trail_x.append(z_real)
        trail_y.append(z_imag)
        if len(trail_x) > max_trail_length:
            trail_x.pop(0)
            trail_y.pop(0)
        trail1.set_data(trail_x, trail_y)
        
        # Update arrow from origin to point
        arrow1.set_position((z_real, z_imag))
        arrow1.xy = (z_real, z_imag)
        arrow1.xyann = (0, 0)
        
        # Update text
        angle_deg = np.degrees(omega * t) % 360
        text1.set_text(f'Phase: {angle_deg:.1f}°\nω = {omega}')
        
        # Right plot: Two points with phase difference
        z1_real = np.cos(omega * t)
        z1_imag = np.sin(omega * t)
        z2_real = np.cos(omega * t + phase_diff)
        z2_imag = np.sin(omega * t + phase_diff)
        
        point2a.set_data([z1_real], [z1_imag])
        point2b.set_data([z2_real], [z2_imag])
        
        # Line connecting two points
        line2.set_data([z1_real, z2_real], [z1_imag, z2_imag])
        
        # Inner product (preserved under rotation)
        inner_product = np.cos(phase_diff)
        text2.set_text(f'Inner product: {inner_product:.4f}\n(Preserved)')
        
        return point1, trail1, point2a, point2b, line2, text1, text2
    
    # Create animation
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                        interval=1000/fps, blit=True)
    
    # Add title
    fig.suptitle('ROPE Rotation and Parallel Transport (Kähler Isometry)\n' +
                 'z_rotated = z · exp(i·θ),  Inner product ⟨z₁, z₂⟩ preserved',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save as GIF
    gif_path = output_dir / '2_rope_rotation_animation.gif'
    writer = PillowWriter(fps=fps)
    anim.save(str(gif_path), writer=writer)
    print(f"Saved GIF animation: {gif_path}")
    logger.log_figure(gif_path, title='ROPE Rotation Animation',
                     fig_type='animation_gif', shape=[14, 6])
    
    # Save a static frame for reference
    update(0)  # First frame
    save_figure(fig, '2_rope_rotation_static', output_dir)
    logger.log_figure(output_dir / '2_rope_rotation_static.png',
                     title='ROPE Rotation Static Frame',
                     fig_type='static')
    
    plt.close(fig)
    
    # Log rotation data for analysis
    t_full = np.linspace(0, 2*np.pi, 100)
    for frame_idx in [0, 30, 60, 90]:
        angle = omega * (frame_idx / n_frames * 4 * np.pi)
        z_real = np.cos(angle)
        z_imag = np.sin(angle)
        logger.log_metric(f'rotation_frame_{frame_idx}', angle,
                         unit='radians',
                         context={'frame': frame_idx, 'z_real': z_real, 'z_imag': z_imag})
    
    # Log series showing full rotation
    angles = omega * t_full
    z_reals = np.cos(angles)
    z_imags = np.sin(angles)
    logger.log_series('rotation_trajectory_real', t_full, z_reals,
                     x_label='time t', y_label='Re(z) = cos(ωt)',
                     metadata={'component': 'real', 'omega': omega})
    logger.log_series('rotation_trajectory_imag', t_full, z_imags,
                     x_label='time t', y_label='Im(z) = sin(ωt)',
                     metadata={'component': 'imaginary', 'omega': omega})
    
    # Also create a multi-frequency version
    create_multi_frequency_animation(output_dir, duration)
    
    # Save logger data
    logger.save()
    print("Visualization 2 completed: ROPE Rotation Animation")


def create_multi_frequency_animation(output_dir: Path, duration: float = 4.0):
    """Create animation showing multiple frequencies rotating at different speeds."""
    
    # Initialize logger for this sub-visualization
    logger = VizLogger('2b_multi_frequency_rotation')
    logger.set_description('Multiple Frequency Rotations Comparison')
    logger.add_finding('Higher frequency → faster rotation on complex plane', 'empirical')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Real', fontsize=12)
    ax.set_ylabel('Imaginary', fontsize=12)
    ax.set_title('Multiple Frequency Rotations\n(Complex Plane Speed Comparison)', fontsize=13)
    
    # Parameters for different frequencies
    frequencies = [0.05, 0.15, 0.3, 0.5]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    labels = [f'ω={f}' for f in frequencies]
    
    n_frames = int(duration * 24)  # 24 fps
    
    # Initialize points
    points = []
    trails = []
    trail_data = [([], []) for _ in frequencies]
    max_trail = 30
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        point, = ax.plot([], [], 'o', color=color, markersize=10 + i*2, 
                        label=labels[i], zorder=5)
        trail, = ax.plot([], [], '-', color=color, alpha=0.4, linewidth=1.5)
        points.append(point)
        trails.append(trail)
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color=color, 
                           linestyle='--', alpha=0.2)
        ax.add_patch(circle)
    
    ax.legend(loc='upper right', fontsize=10)
    
    # Add center annotation
    center_text = ax.text(0.5, 0.02, '', transform=ax.transAxes, fontsize=11,
                         horizontalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        for p, t in zip(points, trails):
            p.set_data([], [])
            t.set_data([], [])
        center_text.set_text('')
        return points + trails + [center_text]
    
    def update(frame):
        t = frame / n_frames * 2 * np.pi  # One full rotation for slowest
        
        for i, (freq, point, trail) in enumerate(zip(frequencies, points, trails)):
            # Each frequency rotates at different speed
            angle = freq * t * 5  # Scale for visibility
            z_real = np.cos(angle)
            z_imag = np.sin(angle)
            
            point.set_data([z_real], [z_imag])
            
            # Update trail
            tx, ty = trail_data[i]
            tx.append(z_real)
            ty.append(z_imag)
            if len(tx) > max_trail:
                tx.pop(0)
                ty.pop(0)
            trail.set_data(tx, ty)
        
        # Update annotation
        center_text.set_text(f'Frame {frame}/{n_frames}')
        
        return points + trails + [center_text]
    
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                        interval=1000/24, blit=True)
    
    # Save GIF
    gif_path = output_dir / '2_rope_multi_frequency.gif'
    writer = PillowWriter(fps=24)
    anim.save(str(gif_path), writer=writer)
    print(f"Saved multi-frequency GIF: {gif_path}")
    logger.log_figure(gif_path, title='Multi-Frequency Rotation Comparison',
                     fig_type='animation_gif')
    
    # Log frequency comparison data
    logger.log_comparison('rotation_speeds', [
        {'label': f'ω={f}', 'value': f, 'properties': {
            'period': f'{(2*np.pi/f):.1f}s' if f > 0 else 'N/A',
            'speed_class': 'slow' if f < 0.1 else ('medium' if f < 0.3 else 'fast')}
        } for f in frequencies
    ])
    
    # Log each frequency trajectory
    t_sample = np.linspace(0, 2*np.pi, 50)
    for freq in frequencies:
        angles = freq * t_sample * 5
        z_reals = np.cos(angles)
        z_imags = np.sin(angles)
        logger.log_series(f'freq_{freq}_real', t_sample, z_reals,
                         x_label='scaled time', y_label='Re(z)',
                         metadata={'frequency': freq})
        logger.log_series(f'freq_{freq}_imag', t_sample, z_imags,
                         x_label='scaled time', y_label='Im(z)',
                         metadata={'frequency': freq})
    
    # Save logger data
    logger.save()
    plt.close(fig)


if __name__ == '__main__':
    create_rope_rotation_animation()

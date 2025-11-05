"""
Quick visualization script to compare BGT60TR13C simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from rcs_simulator import BGT60RCSSimulator, FINGER_GEOMETRIES

def plot_rcs_comparison():
    """Plot RCS comparison across fingers and distances"""
    sim = BGT60RCSSimulator()
    
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    distances = np.linspace(0.15, 1.5, 30)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: RCS vs Distance
    ax = axes[0, 0]
    for finger in fingers:
        rcs_values = []
        for dist in distances:
            result = sim.calculate_finger_rcs(finger, dist)
            rcs_values.append(result['rcs_dbsm'])
        ax.plot(distances, rcs_values, marker='o', label=finger, linewidth=2)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('RCS (dBsm)', fontsize=12)
    ax.set_title('RCS vs Distance for Different Fingers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: SNR vs Distance
    ax = axes[0, 1]
    for finger in fingers:
        snr_values = []
        for dist in distances:
            result = sim.calculate_finger_rcs(finger, dist)
            snr_values.append(result['snr_db'])
        ax.plot(distances, snr_values, marker='o', label=finger, linewidth=2)
    
    ax.axhline(y=10, color='r', linestyle='--', label='Detection Threshold (10 dB)')
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('SNR vs Distance (BGT60TR13C)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Range-Doppler Map Example
    ax = axes[1, 0]
    rd_map, metadata = sim.generate_range_doppler_signature(
        'index', distance_m=0.3, velocity_ms=0.2
    )
    im = ax.imshow(rd_map, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel('Range Bin', fontsize=12)
    ax.set_ylabel('Doppler Bin', fontsize=12)
    ax.set_title('Range-Doppler Map (Index Finger, 0.3m, 0.2 m/s)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=metadata['target_range_bin'], color='r', linestyle='--', alpha=0.7)
    ax.axhline(y=metadata['target_doppler_bin'], color='r', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    # Plot 4: Finger Geometry Comparison
    ax = axes[1, 1]
    finger_names = list(FINGER_GEOMETRIES.keys())
    lengths = [FINGER_GEOMETRIES[f].length for f in finger_names]
    diameters = [FINGER_GEOMETRIES[f].diameter for f in finger_names]
    
    x = np.arange(len(finger_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, lengths, width, label='Length (mm)', alpha=0.8)
    bars2 = ax.bar(x + width/2, diameters, width, label='Diameter (mm)', alpha=0.8)
    
    ax.set_xlabel('Finger Type', fontsize=12)
    ax.set_ylabel('Dimension (mm)', fontsize=12)
    ax.set_title('Finger Geometry Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(finger_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('bgt60_simulation_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: bgt60_simulation_results.png")
    plt.show()


def print_detection_matrix():
    """Print detection capability matrix"""
    sim = BGT60RCSSimulator()
    
    print("\n" + "="*70)
    print("BGT60TR13C DETECTION CAPABILITY MATRIX")
    print("="*70)
    print("SNR Threshold: 10 dB for reliable detection\n")
    
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    distances = [0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    
    # Header
    print(f"{'Finger':<10}", end='')
    for dist in distances:
        print(f"{dist:>8.2f}m", end='')
    print()
    print("-" * 70)
    
    # Data rows
    for finger in fingers:
        print(f"{finger:<10}", end='')
        for dist in distances:
            result = sim.calculate_finger_rcs(finger, dist)
            snr = result['snr_db']
            if snr > 20:
                symbol = "  ✓✓  "  # Excellent
            elif snr > 10:
                symbol = "  ✓   "  # Good
            elif snr > 5:
                symbol = "  ~   "  # Marginal
            else:
                symbol = "  ✗   "  # Poor
            print(f"{symbol}", end='')
        print()
    
    print("\nLegend: ✓✓ = Excellent (>20dB), ✓ = Good (>10dB), ~ = Marginal (>5dB), ✗ = Poor (<5dB)")
    print("="*70 + "\n")


def compare_tissue_properties():
    """Show tissue properties at 60 GHz"""
    sim = BGT60RCSSimulator()
    
    print("\n" + "="*70)
    print("HUMAN TISSUE ELECTROMAGNETIC PROPERTIES AT 60 GHz")
    print("="*70)
    print(f"Operating Frequency: {sim.config.frequency_center/1e9:.2f} GHz")
    print(f"Wavelength in free space: {sim.config.wavelength*1000:.2f} mm")
    print(f"\nComplex Permittivity (ε_r):")
    print(f"  Real part (ε'): {sim.epsilon_r.real:.1f}")
    print(f"  Imaginary part (ε''): {sim.epsilon_r.imag:.1f}")
    print(f"\nSkin Depth (δ): {sim.skin_depth:.2f} mm")
    print(f"  → mmWave radiation penetrates only {sim.skin_depth:.2f} mm into tissue")
    print(f"  → Surface/skin scattering dominates at 60 GHz")
    
    print(f"\nReflection Coefficient: ", end='')
    reflection = abs((np.sqrt(sim.epsilon_r) - 1) / (np.sqrt(sim.epsilon_r) + 1))**2
    print(f"{reflection:.3f} ({10*np.log10(reflection):.1f} dB)")
    
    print("\nImplication for Finger Detection:")
    print("  - High surface reflection enables good detection")
    print("  - Minimal penetration reduces body effects")
    print("  - Small features (rings, wrinkles) create scattering")
    print("  - Sensitive to skin moisture and temperature")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("BGT60TR13C Simulation Analysis")
    print("=" * 70)
    
    # Print detection matrix
    print_detection_matrix()
    
    # Print tissue properties
    compare_tissue_properties()
    
    # Generate plots
    print("Generating visualization plots...")
    plot_rcs_comparison()
    
    print("\nAnalysis complete!")

"""
RCS (Radar Cross Section) Simulator for Hand Finger Detection
Specifically designed for Infineon BGT60TR13C 60 GHz FMCW Radar Module

This module simulates the radar cross section of individual fingers using
electromagnetic scattering theory adapted for 60 GHz millimeter-wave radar.

BGT60TR13C Specifications:
- Frequency Range: 58.0-63.5 GHz (center ~60.75 GHz)
- Bandwidth: Up to 5.5 GHz
- Wavelength: ~4.94 mm at 60.75 GHz
- 1 TX, 3 RX channels
- 12-bit ADC, up to 4 MSps sampling
- Range Resolution: ~2.73 cm (with 5.5 GHz BW)
- Maximum Range: ~5m (typical for gesture detection)
- FMCW chirp modulation
"""

import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class BGT60TR13CConfig:
    """Configuration for BGT60TR13C radar parameters"""
    # RF Frontend specifications
    frequency_min: float = 58.0e9  # Hz
    frequency_max: float = 63.5e9  # Hz
    frequency_center: float = 60.75e9  # Hz (center frequency)
    bandwidth: float = 5.5e9  # Hz (maximum usable bandwidth)
    
    # Transmitter
    tx_power_dbm: float = 11.0  # dBm (typical EIRP)
    num_tx: int = 1  # 1 TX channel
    
    # Receiver
    num_rx: int = 3  # 3 RX channels
    rx_noise_figure: float = 15.0  # dB (typical NF)
    
    # ADC specifications
    adc_bits: int = 12  # 12-bit resolution
    sampling_rate_max: float = 4e6  # 4 MSps
    
    # Antenna (integrated in package)
    antenna_gain_tx: float = 8.0  # dBi (typical integrated antenna)
    antenna_gain_rx: float = 8.0  # dBi
    antenna_beamwidth: float = 80.0  # degrees (typical azimuth beamwidth)
    
    # Derived parameters
    wavelength: float = None
    range_resolution: float = None
    max_unambiguous_range: float = 5.0  # meters (typical for gesture detection)
    velocity_resolution: float = None
    
    def __post_init__(self):
        c = 3e8  # speed of light (m/s)
        if self.wavelength is None:
            self.wavelength = c / self.frequency_center  # ~4.94 mm at 60.75 GHz
        if self.range_resolution is None:
            self.range_resolution = c / (2 * self.bandwidth)  # ~2.73 cm with 5.5 GHz BW


@dataclass
class FingerGeometry:
    """Physical dimensions and properties of a finger"""
    length: float  # mm
    diameter: float  # mm
    name: str
    
    @property
    def radius(self) -> float:
        """Radius in mm"""
        return self.diameter / 2.0
    
    @property
    def cross_sectional_area(self) -> float:
        """Cross-sectional area in mm²"""
        return np.pi * (self.radius ** 2)


# Empirical finger dimensions (adult human average)
FINGER_GEOMETRIES = {
    'thumb': FingerGeometry(length=48.0, diameter=20.0, name='thumb'),
    'index': FingerGeometry(length=70.0, diameter=16.0, name='index'),
    'middle': FingerGeometry(length=76.0, diameter=16.5, name='middle'),
    'ring': FingerGeometry(length=69.0, diameter=15.5, name='ring'),
    'pinky': FingerGeometry(length=58.0, diameter=13.5, name='pinky'),
}


class BGT60RCSSimulator:
    """
    Simulate RCS signatures for human fingers using the BGT60TR13C radar.
    
    This simulator models:
    1. FMCW radar signal processing
    2. Electromagnetic scattering at 60 GHz
    3. Human tissue dielectric properties at mmWave frequencies
    4. Multi-path and clutter effects
    """
    
    def __init__(self, config: BGT60TR13CConfig = None):
        """
        Initialize RCS simulator for BGT60TR13C.
        
        Args:
            config: Radar configuration (defaults to BGT60TR13C specs)
        """
        self.config = config or BGT60TR13CConfig()
        
        # Physical constants
        self.c = 3e8  # speed of light (m/s)
        self.k = 2 * np.pi / self.config.wavelength  # wavenumber (rad/m)
        
        # Human tissue properties at 60 GHz
        # Based on ITU-R P.2040 and recent mmWave tissue studies
        self.epsilon_r = self._get_tissue_permittivity_60ghz()
        self.skin_depth = self._calculate_skin_depth()
        
    def _get_tissue_permittivity_60ghz(self) -> complex:
        """
        Complex permittivity of human skin/tissue at 60 GHz.
        
        At 60 GHz, human tissue has high water content influence:
        - Real part (ε'): ~10-12 (relative permittivity)
        - Imaginary part (ε''): ~15-20 (loss factor)
        
        Returns:
            Complex permittivity (ε_r)
        """
        epsilon_real = 11.0  # relative permittivity at 60 GHz
        epsilon_imag = 17.0  # loss factor at 60 GHz
        return complex(epsilon_real, epsilon_imag)
    
    def _calculate_skin_depth(self) -> float:
        """
        Calculate skin depth (penetration depth) in human tissue at 60 GHz.
        
        Skin depth = 1 / (2 * α), where α is attenuation constant
        
        Returns:
            Skin depth in meters (typically ~0.4-0.6 mm at 60 GHz)
        """
        epsilon_0 = 8.854e-12  # F/m
        mu_0 = 4 * np.pi * 1e-7  # H/m
        
        epsilon_real = self.epsilon_r.real
        epsilon_imag = self.epsilon_r.imag
        
        omega = 2 * np.pi * self.config.frequency_center
        
        # Attenuation constant (simplified)
        alpha = omega * np.sqrt(mu_0 * epsilon_0 * epsilon_real / 2) * \
                np.sqrt(np.sqrt(1 + (epsilon_imag/epsilon_real)**2) - 1)
        
        skin_depth = 1 / alpha  # meters
        return skin_depth * 1000  # convert to mm
    
    def calculate_cylinder_rcs(
        self,
        length_mm: float,
        diameter_mm: float,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
        polarization: str = 'VV'
    ) -> float:
        """
        Calculate RCS of a cylindrical object (finger model) at 60 GHz.
        
        Uses Physical Optics (PO) approximation for high-frequency scattering.
        At 60 GHz (λ ≈ 5 mm), fingers are electrically large objects.
        
        Args:
            length_mm: Cylinder length in mm
            diameter_mm: Cylinder diameter in mm
            azimuth_deg: Azimuth angle in degrees (0 = broadside)
            elevation_deg: Elevation angle in degrees
            polarization: 'VV' (vertical) or 'HH' (horizontal)
        
        Returns:
            RCS in m² (linear scale)
        """
        # Convert to meters and radians
        length = length_mm / 1000.0
        radius = (diameter_mm / 2.0) / 1000.0
        azimuth = np.deg2rad(azimuth_deg)
        elevation = np.deg2rad(elevation_deg)
        
        # Effective illuminated area depends on angle
        theta = azimuth  # angle from broadside
        
        # For cylinder at 60 GHz (high frequency regime)
        if abs(theta) < np.deg2rad(5):  # Near-broadside
            # Broadside: RCS ≈ (4π/λ²) * A² where A is physical area
            # For dielectric cylinder with ε_r ≈ 11:
            area = length * 2 * radius  # projected area
            reflection_coeff = abs((np.sqrt(self.epsilon_r) - 1) / 
                                  (np.sqrt(self.epsilon_r) + 1))**2
            rcs = (4 * np.pi / self.config.wavelength**2) * area**2 * reflection_coeff
            
        elif abs(theta) > np.deg2rad(85):  # End-on
            # End-on: circular cross-section
            area = np.pi * radius**2
            rcs = (4 * np.pi / self.config.wavelength**2) * area**2 * 0.5
            
        else:  # Intermediate angles
            # Creeping wave and specular reflection
            # RCS varies as cos²(θ) for cylinder
            broadside_area = length * 2 * radius
            effective_area = broadside_area * abs(np.cos(theta))
            
            reflection_coeff = abs((np.sqrt(self.epsilon_r) - 1) / 
                                  (np.sqrt(self.epsilon_r) + 1))**2
            
            rcs = (4 * np.pi / self.config.wavelength**2) * \
                  effective_area**2 * reflection_coeff * abs(np.cos(theta))
        
        # Add diffraction effects for edges (significant at 60 GHz)
        edge_contribution = (radius / self.config.wavelength) * 1e-5
        rcs += edge_contribution
        
        # Ensure minimum RCS (noise floor consideration)
        min_rcs = 1e-6  # m² (-60 dBsm)
        return max(rcs, min_rcs)
    
    def calculate_finger_rcs(
        self,
        finger_type: str,
        distance_m: float,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
        flexion_deg: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate RCS for a specific finger type.
        
        Args:
            finger_type: One of 'thumb', 'index', 'middle', 'ring', 'pinky'
            distance_m: Distance from radar in meters
            azimuth_deg: Azimuth angle in degrees
            elevation_deg: Elevation angle in degrees
            flexion_deg: Finger flexion/curl angle (0=straight, 90=fully curled)
        
        Returns:
            Dictionary with RCS values and parameters
        """
        if finger_type not in FINGER_GEOMETRIES:
            raise ValueError(f"Unknown finger type: {finger_type}")
        
        finger = FINGER_GEOMETRIES[finger_type]
        
        # Effective length changes with flexion
        effective_length = finger.length * np.cos(np.deg2rad(flexion_deg / 2))
        
        # Calculate base RCS
        rcs_linear = self.calculate_cylinder_rcs(
            length_mm=effective_length,
            diameter_mm=finger.diameter,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg
        )
        
        # Path loss at 60 GHz (includes atmospheric absorption)
        # At 60 GHz, oxygen absorption is significant (~15 dB/km)
        path_loss_db = 20 * np.log10(distance_m) + \
                       20 * np.log10(self.config.frequency_center) - \
                       20 * np.log10(self.c) + \
                       0.015 * distance_m  # oxygen absorption (dB/m at 60 GHz)
        
        # Received power using radar equation
        # P_r = (P_t * G_t * G_r * λ² * σ) / ((4π)³ * R⁴)
        tx_power_w = 10**(self.config.tx_power_dbm / 10) / 1000  # Watts
        gain_linear = 10**(self.config.antenna_gain_tx / 10) * \
                     10**(self.config.antenna_gain_rx / 10)
        
        received_power_w = (tx_power_w * gain_linear * 
                           self.config.wavelength**2 * rcs_linear) / \
                          ((4 * np.pi)**3 * distance_m**4)
        
        # Convert to dBm
        received_power_dbm = 10 * np.log10(received_power_w * 1000)
        
        # SNR calculation (simplified)
        # Thermal noise: -174 dBm/Hz + 10*log10(BW) + NF
        noise_power_dbm = -174 + 10 * np.log10(self.config.bandwidth) + \
                         self.config.rx_noise_figure
        snr_db = received_power_dbm - noise_power_dbm
        
        return {
            'finger_type': finger_type,
            'distance_m': distance_m,
            'azimuth_deg': azimuth_deg,
            'elevation_deg': elevation_deg,
            'flexion_deg': flexion_deg,
            'rcs_linear_m2': rcs_linear,
            'rcs_dbsm': 10 * np.log10(rcs_linear),
            'received_power_dbm': received_power_dbm,
            'snr_db': snr_db,
            'path_loss_db': path_loss_db,
            'detectable': snr_db > 10.0  # 10 dB SNR threshold
        }
    
    def generate_range_doppler_signature(
        self,
        finger_type: str,
        distance_m: float,
        velocity_ms: float = 0.0,
        num_chirps: int = 128,
        num_samples: int = 256
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate simulated Range-Doppler map for a finger.
        
        This simulates the output you'd get from BGT60TR13C FMCW processing.
        
        Args:
            finger_type: Finger type
            distance_m: Distance in meters
            velocity_ms: Radial velocity in m/s (positive = approaching)
            num_chirps: Number of chirps in frame (time dimension)
            num_samples: Number of ADC samples per chirp (range dimension)
        
        Returns:
            Tuple of (range_doppler_map, metadata)
        """
        # Calculate RCS
        rcs_data = self.calculate_finger_rcs(finger_type, distance_m)
        
        # Range bin calculation
        range_bin = int((distance_m / self.config.max_unambiguous_range) * num_samples)
        range_bin = np.clip(range_bin, 0, num_samples - 1)
        
        # Doppler frequency calculation
        # f_d = 2 * v * f_c / c
        doppler_freq = 2 * velocity_ms * self.config.frequency_center / self.c
        
        # Doppler bin (assuming PRF allows this velocity)
        # For 60 GHz, max unambiguous velocity is typically ±1-2 m/s for gesture detection
        max_velocity = 2.0  # m/s
        doppler_bin = int((velocity_ms / max_velocity + 1) * num_chirps / 2)
        doppler_bin = np.clip(doppler_bin, 0, num_chirps - 1)
        
        # Create Range-Doppler map
        rd_map = np.random.randn(num_chirps, num_samples) * 0.1  # Noise floor
        
        # Add target peak
        if rcs_data['detectable']:
            signal_amplitude = np.sqrt(10**(rcs_data['snr_db'] / 10))
            
            # Add Gaussian spread around target location
            for i in range(-2, 3):
                for j in range(-2, 3):
                    r_idx = range_bin + i
                    d_idx = doppler_bin + j
                    if 0 <= r_idx < num_samples and 0 <= d_idx < num_chirps:
                        spread = np.exp(-(i**2 + j**2) / 2)
                        rd_map[d_idx, r_idx] += signal_amplitude * spread
        
        metadata = {
            'range_resolution_m': self.config.range_resolution,
            'velocity_resolution_ms': max_velocity / num_chirps,
            'target_range_bin': range_bin,
            'target_doppler_bin': doppler_bin,
            'rcs_data': rcs_data
        }
        
        return rd_map, metadata
    
    def simulate_multi_finger_scenario(
        self,
        finger_configs: List[Dict],
        num_chirps: int = 128,
        num_samples: int = 256
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Simulate multiple fingers in the radar field of view.
        
        Args:
            finger_configs: List of dicts with keys: 'type', 'distance', 'velocity', 
                           'azimuth', 'elevation', 'flexion'
            num_chirps: Number of chirps
            num_samples: Number of samples per chirp
        
        Returns:
            Combined Range-Doppler map and list of metadata for each finger
        """
        rd_map_combined = np.random.randn(num_chirps, num_samples) * 0.1
        all_metadata = []
        
        for config in finger_configs:
            rd_map, metadata = self.generate_range_doppler_signature(
                finger_type=config.get('type', 'index'),
                distance_m=config.get('distance', 0.3),
                velocity_ms=config.get('velocity', 0.0),
                num_chirps=num_chirps,
                num_samples=num_samples
            )
            
            # Superpose signals (linear addition in complex domain, simplified here)
            rd_map_combined += rd_map
            all_metadata.append(metadata)
        
        return rd_map_combined, all_metadata


def main():
    """Example usage of BGT60RCSSimulator"""
    print("BGT60TR13C RCS Simulator Demo\n" + "="*50)
    
    # Initialize simulator
    sim = BGT60RCSSimulator()
    
    print(f"\nRadar Configuration:")
    print(f"  Frequency: {sim.config.frequency_center/1e9:.2f} GHz")
    print(f"  Bandwidth: {sim.config.bandwidth/1e9:.2f} GHz")
    print(f"  Wavelength: {sim.config.wavelength*1000:.2f} mm")
    print(f"  Range Resolution: {sim.config.range_resolution*100:.2f} cm")
    print(f"  Skin Depth: {sim.skin_depth:.2f} mm")
    
    # Test different fingers
    print(f"\n{'Finger':<10} {'Distance':<10} {'RCS (dBsm)':<12} {'Rx Power':<12} {'SNR (dB)':<10} {'Detectable'}")
    print("-" * 80)
    
    for finger_type in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        for distance in [0.2, 0.5, 1.0]:
            result = sim.calculate_finger_rcs(finger_type, distance)
            print(f"{finger_type:<10} {distance:<10.2f} {result['rcs_dbsm']:<12.2f} "
                  f"{result['received_power_dbm']:<12.2f} {result['snr_db']:<10.2f} "
                  f"{'Yes' if result['detectable'] else 'No'}")
    
    # Generate Range-Doppler map
    print("\nGenerating Range-Doppler map for index finger at 0.3m...")
    rd_map, metadata = sim.generate_range_doppler_signature('index', 0.3, velocity_ms=0.1)
    print(f"  Shape: {rd_map.shape}")
    print(f"  Target at range bin: {metadata['target_range_bin']}")
    print(f"  Target at Doppler bin: {metadata['target_doppler_bin']}")
    print(f"  Peak SNR: {np.max(rd_map):.2f}")


if __name__ == "__main__":
    main()

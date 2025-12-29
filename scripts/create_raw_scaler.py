"""
Create a scaler that normalizes raw extracted features to match
the standardized distribution expected by the model.
"""
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Typical ranges for raw audio features (estimated from librosa outputs)
# These are approximate mean and std values for raw features
RAW_FEATURE_STATS = {
    'duration': (200, 100),  # seconds
    'tempo': (120, 30),      # BPM
    'key': (5, 3.5),         # 0-11
    'mode': (0.5, 0.5),      # 0 or 1
    'time_signature': (4, 1), # typically 3-4
    'rhythmic_complexity': (4, 2),
    'spectral_centroid_mean': (0, 50),
    'spectral_rolloff_mean': (0, 100),
    'mfcc1': (0, 50),
    'mfcc2': (0, 50),
    'mfcc3': (0, 30),
    'timbre_std': (15, 10),
    'timbre_skew': (0, 1),
    'timbre_kurtosis': (1, 2),
    'spec_bw': (2000, 1000),
    'zcr': (0.1, 0.2),
    'rms': (50, 50),
    'timbre_max': (50, 30),
    'timbre_min': (-50, 30),
    'timbre_range': (100, 50),
    'timbre_median': (5, 10),
    'timbre_variance': (500, 300),
    'timbre_mad': (3, 2),
    'timbre_temporal_centroid': (0.5, 0.2),
    'timbre_q1': (0, 5),
    'timbre_q3': (10, 10),
    'timbre_iqr': (15, 10),
    'timbre_entropy': (10, 2),
    'timbre_energy': (50000000, 30000000),
    'timbre_flux': (25, 15),
    'timbre_flatness': (0.7, 0.3),
    'num_non_zero_rms_segments': (50, 30),
    'brightness': (1, 0.5),
    'loudness_mean': (0.2, 0.1),
    'loudness_std': (0.08, 0.05),
    'loudness_skew': (0, 1),
    'loudness_kurtosis': (0, 2),
    'loudness_max': (0.4, 0.2),
    'loudness_min': (0, 0.1),
    'loudness_range': (0.4, 0.2),
    'loudness_median': (0.2, 0.1),
    'loudness_q1': (0.15, 0.1),
    'loudness_q3': (0.25, 0.1),
    'loudness_iqr': (0.1, 0.05),
    'loudness_entropy': (10, 1),
    'loudness_energy': (1000, 500),
    'loudness_flux': (0.001, 0.001),
    'roughness': (0.02, 0.02),
    'chroma_mean': (0.5, 0.2),
    'chroma_std': (0.2, 0.1),
    'chroma_skew': (0.2, 0.5),
    'chroma_kurtosis': (0, 1),
    'chroma_max': (0.7, 0.2),
    'chroma_min': (0.4, 0.2),
    'chroma_range': (0.2, 0.1),
    'chroma_median': (0.5, 0.2),
    'chroma_q1': (0.4, 0.2),
    'chroma_q3': (0.7, 0.2),
    'chroma_iqr': (0.3, 0.2),
    'chroma_entropy': (10, 1),
    'chroma_energy': (8000, 4000),
    'chroma_flux': (0.001, 0.001),
    'chroma_flatness': (0.9, 0.1),
    'melodic_contour_direction': (0, 0.5),
    'melodic_contour_interval': (0.3, 0.2),
    'rhythmic_entropy': (8, 2),
    'rhythmic_irregularity': (0.5, 0.3),
}

# Get ordered list of features matching what extract_features returns
FEATURE_ORDER = list(RAW_FEATURE_STATS.keys())

def create_raw_scaler():
    """Create a scaler that normalizes raw features."""
    # Build mean and scale arrays
    means = np.array([RAW_FEATURE_STATS[f][0] for f in FEATURE_ORDER])
    stds = np.array([RAW_FEATURE_STATS[f][1] for f in FEATURE_ORDER])

    # Create a StandardScaler with pre-set parameters
    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.scale_ = stds
    scaler.var_ = stds ** 2
    scaler.n_features_in_ = len(FEATURE_ORDER)
    scaler.n_samples_seen_ = 1000  # dummy value

    return scaler, FEATURE_ORDER

if __name__ == '__main__':
    scaler, feature_order = create_raw_scaler()

    # Save the scaler
    joblib.dump(scaler, 'raw_scaler.pkl')
    joblib.dump(feature_order, 'feature_order.pkl')

    print(f"Created raw_scaler.pkl with {len(feature_order)} features")
    print("Feature order:", feature_order[:5], "...")

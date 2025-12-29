"""Debug feature extraction pipeline."""

import os
import sys
import joblib
import numpy as np
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from optimized_datasets import gtzan_to_h5
from optimized_datasets.feature_extraction import extract_features
from nn_genre_guesser.genre_guesser import FEATURE_ORDER, flatten_dict_ordered

def debug_file(file_path):
    """Debug feature extraction for a file."""
    print(f"\n{'='*70}")
    print(f"Debugging: {file_path}")
    print(f"{'='*70}")

    # Convert to h5
    file_name = file_path.rsplit('.', 1)[0]
    h5_file_name = file_name + '.h5'

    print(f"\n1. Converting to H5: {h5_file_name}")
    gtzan_to_h5.convert_gtzan_to_msd_structure(file_path, h5_file_name)

    # Extract features
    print(f"\n2. Extracting features...")
    with h5py.File(h5_file_name, 'r') as f:
        features_dict = extract_features(f)

    print(f"   Extracted {len(features_dict)} features")

    # Show first 10 features
    print("\n3. Raw feature values (first 10):")
    for i, key in enumerate(FEATURE_ORDER[:10]):
        val = features_dict.get(key, 'MISSING')
        print(f"   {key}: {val}")

    # Check for missing features
    missing = [k for k in FEATURE_ORDER if k not in features_dict]
    if missing:
        print(f"\n   MISSING FEATURES: {missing}")

    # Flatten to ordered array
    features_array = np.array(flatten_dict_ordered(features_dict), dtype=float).reshape(1, -1)
    print(f"\n4. Feature array shape: {features_array.shape}")
    print(f"   Min: {features_array.min():.4f}, Max: {features_array.max():.4f}")
    print(f"   Mean: {features_array.mean():.4f}, Std: {features_array.std():.4f}")

    # Load scaler and apply
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')

    print(f"\n5. Applying scaler...")
    features_scaled = scaler.transform(features_array)
    print(f"   Scaled Min: {features_scaled.min():.4f}, Max: {features_scaled.max():.4f}")
    print(f"   Scaled Mean: {features_scaled.mean():.4f}, Std: {features_scaled.std():.4f}")

    # Check for extreme values
    extreme_idx = np.where(np.abs(features_scaled) > 10)[1]
    if len(extreme_idx) > 0:
        print(f"\n   WARNING: Extreme values at indices: {extreme_idx[:5]}")
        for idx in extreme_idx[:5]:
            print(f"      {FEATURE_ORDER[idx]}: {features_scaled[0, idx]:.2f}")

    print(f"\n6. Applying selector...")
    features_selected = selector.transform(features_scaled)
    print(f"   Selected shape: {features_selected.shape}")
    print(f"   Selected Min: {features_selected.min():.4f}, Max: {features_selected.max():.4f}")

    # Reshape for CNN
    features_3d = features_selected.reshape((1, features_selected.shape[1], 1))
    print(f"\n7. Final input shape: {features_3d.shape}")

    # Load model and predict
    from keras.models import load_model
    model = load_model('trained_model.keras')
    prediction = model.predict(features_3d, verbose=0)

    print(f"\n8. Raw prediction output:")
    label_encoder = joblib.load('label_encoder.pkl')
    for i, (genre, prob) in enumerate(zip(label_encoder.classes_, prediction[0])):
        if prob > 0.01:
            print(f"   {genre}: {prob:.4f}")

    return prediction


if __name__ == "__main__":
    # Test with Nirvana
    debug_file("ui/nirvana_smells_like_teen_spirit.wav")

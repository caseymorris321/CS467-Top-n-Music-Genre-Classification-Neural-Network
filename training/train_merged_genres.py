"""
Train with merged genres for better real-world performance.
Merges similar genres like rock/pop_rock/metal, hiphop/rap, etc.
"""

import os
import h5py
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from optimized_datasets.feature_extraction import extract_features
from nn_genre_guesser.genre_guesser import FEATURE_ORDER, flatten_dict_ordered


# Merge similar genres
GENRE_MAPPING = {
    'blues': 'blues',
    'classical': 'classical',
    'country': 'country',
    'disco': 'electronic',  # merge into electronic
    'electronic': 'electronic',
    'hiphop': 'rap',  # merge hiphop into rap
    'international': 'international',
    'jazz': 'jazz',
    'latin': 'latin',
    'metal': 'metal',
    'new age': 'new age',
    'pop': 'pop',
    'pop_rock': 'rock',  # merge pop_rock into rock
    'rap': 'rap',
    'reggae': 'reggae',
    'rnb': 'rnb',
    'rock': 'rock',
}


def find_h5_files(directory):
    """Find all H5 files in directory."""
    return [os.path.join(root, f) for root, _, files in os.walk(directory)
            for f in files if f.endswith('.h5')]


def extract_dataset(h5_dir):
    """Extract features from all H5 files with genre merging."""
    h5_files = find_h5_files(h5_dir)
    print(f"Found {len(h5_files)} H5 files")

    data = []
    for filepath in tqdm(h5_files, desc="Extracting features"):
        try:
            with h5py.File(filepath, 'r') as f:
                genre = f.attrs.get('genre', None)
                if genre is None:
                    continue

                genre = genre.decode() if isinstance(genre, bytes) else str(genre)

                # Map to merged genre
                genre = GENRE_MAPPING.get(genre.lower(), genre.lower())

                features_dict = extract_features(f)
                features_array = flatten_dict_ordered(features_dict)

                data.append({
                    'genre': genre,
                    'features': features_array
                })
        except Exception as e:
            continue

    return data


def train_merged(h5_dir, output_dir='.'):
    """Train model with merged genres."""
    print("=" * 60)
    print("Training with Merged Genres")
    print("=" * 60)

    # Extract features
    print("\n1. Extracting features...")
    data = extract_dataset(h5_dir)
    print(f"   Extracted {len(data)} samples")

    # Convert to arrays
    X = np.array([d['features'] for d in data])
    y = np.array([d['genre'] for d in data])

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Get genre counts
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n   Genre distribution (merged):")
    for g, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"      {g}: {c}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_genres = len(label_encoder.classes_)
    print(f"\n   Number of genres: {num_genres}")
    print(f"   Classes: {list(label_encoder.classes_)}")

    # Scale features
    print("\n2. Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    k_features = min(50, X_scaled.shape[1])
    print(f"\n3. Selecting top {k_features} features...")
    selector = SelectKBest(f_classif, k=k_features)
    X_selected = selector.fit_transform(X_scaled, y_encoded)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\n4. Data split: Train={len(X_train)}, Test={len(X_test)}")

    # Train Random Forest with optimized parameters
    print("\n5. Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"   Random Forest Accuracy: {rf_acc:.2%}")

    # Classification report
    y_pred = rf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    print("\nPer-class accuracy:")
    for i, genre in enumerate(label_encoder.classes_):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {genre}: {class_acc:.2%}")

    # Save
    print("\n6. Saving models...")
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(selector, f'{output_dir}/selector.pkl')
    joblib.dump(rf, f'{output_dir}/random_forest.pkl')

    # Save genre mapping for reference
    joblib.dump(GENRE_MAPPING, f'{output_dir}/genre_mapping.pkl')

    print(f"\nSaved to {output_dir}/")
    return rf, rf_acc


if __name__ == '__main__':
    train_merged('processed_h5', '.')

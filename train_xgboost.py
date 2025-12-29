"""
Train with XGBoost for potentially better accuracy.
"""

import os
import h5py
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from optimized_datasets.feature_extraction import extract_features
from nn_genre_guesser.genre_guesser import FEATURE_ORDER, flatten_dict_ordered

# Merge similar genres
GENRE_MAPPING = {
    'blues': 'blues',
    'classical': 'classical',
    'country': 'country',
    'disco': 'electronic',
    'electronic': 'electronic',
    'hiphop': 'rap',
    'international': 'international',
    'jazz': 'jazz',
    'latin': 'latin',
    'metal': 'metal',
    'new age': 'new age',
    'pop': 'pop',
    'pop_rock': 'rock',
    'rap': 'rap',
    'reggae': 'reggae',
    'rnb': 'rnb',
    'rock': 'rock',
}


def find_h5_files(directory):
    return [os.path.join(root, f) for root, _, files in os.walk(directory)
            for f in files if f.endswith('.h5')]


def extract_dataset(h5_dir):
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
                genre = GENRE_MAPPING.get(genre.lower(), genre.lower())
                features_dict = extract_features(f)
                features_array = flatten_dict_ordered(features_dict)
                data.append({'genre': genre, 'features': features_array})
        except:
            continue
    return data


def train_xgboost(h5_dir, output_dir='.'):
    print("=" * 60)
    print("Training with XGBoost + Ensemble")
    print("=" * 60)

    # Extract features
    print("\n1. Extracting features...")
    data = extract_dataset(h5_dir)
    print(f"   Extracted {len(data)} samples")

    X = np.array([d['features'] for d in data])
    y = np.array([d['genre'] for d in data])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Genre distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n   Genres: {len(unique)}")
    for g, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"      {g}: {c}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_genres = len(label_encoder.classes_)

    # Scale
    print("\n2. Scaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection - use more features
    k_features = min(60, X_scaled.shape[1])
    print(f"\n3. Selecting top {k_features} features...")
    selector = SelectKBest(f_classif, k=k_features)
    X_selected = selector.fit_transform(X_scaled, y_encoded)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"\n4. Train={len(X_train)}, Test={len(X_test)}")

    # Train XGBoost
    print("\n5. Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_acc = xgb.score(X_test, y_test)
    print(f"   XGBoost Accuracy: {xgb_acc:.2%}")

    # Train Random Forest for comparison
    print("\n6. Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"   Random Forest Accuracy: {rf_acc:.2%}")

    # Train Gradient Boosting
    print("\n7. Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_acc = gb.score(X_test, y_test)
    print(f"   Gradient Boosting Accuracy: {gb_acc:.2%}")

    # Ensemble with soft voting
    print("\n8. Creating Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[xgb_acc, rf_acc, gb_acc]  # Weight by accuracy
    )
    ensemble.fit(X_train, y_train)
    ensemble_acc = ensemble.score(X_test, y_test)
    print(f"   Ensemble Accuracy: {ensemble_acc:.2%}")

    # Find best model
    accuracies = {'XGBoost': xgb_acc, 'Random Forest': rf_acc,
                  'Gradient Boosting': gb_acc, 'Ensemble': ensemble_acc}
    best_name = max(accuracies, key=accuracies.get)
    best_acc = accuracies[best_name]

    print(f"\n{'='*60}")
    print("RESULTS:")
    for name, acc in accuracies.items():
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name}: {acc:.2%}{marker}")
    print(f"{'='*60}")

    # Use best model for detailed report
    if best_name == 'XGBoost':
        best_model = xgb
    elif best_name == 'Random Forest':
        best_model = rf
    elif best_name == 'Gradient Boosting':
        best_model = gb
    else:
        best_model = ensemble

    y_pred = best_model.predict(X_test)
    print(f"\n{best_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    print("\nPer-class accuracy:")
    for i, genre in enumerate(label_encoder.classes_):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {genre}: {class_acc:.2%}")

    # Save best model as random_forest.pkl for compatibility
    print("\n9. Saving models...")
    joblib.dump(best_model, f'{output_dir}/random_forest.pkl')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(selector, f'{output_dir}/selector.pkl')

    # Also save individual models
    joblib.dump(xgb, f'{output_dir}/xgboost.pkl')
    joblib.dump(ensemble, f'{output_dir}/ensemble.pkl')

    print(f"\nBest model ({best_name}) saved as random_forest.pkl")
    print(f"Test Accuracy: {best_acc:.2%}")

    return best_model, best_acc


if __name__ == '__main__':
    train_xgboost('processed_h5', '.')

"""
Train directly from H5 files to ensure feature extraction matches inference.
This creates a proper scaler that will work during inference.
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Import the exact same feature extraction used in inference
from optimized_datasets.feature_extraction import extract_features
from nn_genre_guesser.genre_guesser import FEATURE_ORDER, flatten_dict_ordered


def find_h5_files(directory):
    """Find all H5 files in directory."""
    return [os.path.join(root, f) for root, _, files in os.walk(directory)
            for f in files if f.endswith('.h5')]


def extract_dataset(h5_dir):
    """Extract features from all H5 files."""
    h5_files = find_h5_files(h5_dir)
    print(f"Found {len(h5_files)} H5 files")

    data = []
    for filepath in tqdm(h5_files, desc="Extracting features"):
        try:
            with h5py.File(filepath, 'r') as f:
                genre = f.attrs.get('genre', None)
                if genre is None:
                    continue

                features_dict = extract_features(f)
                features_array = flatten_dict_ordered(features_dict)

                data.append({
                    'genre': genre.decode() if isinstance(genre, bytes) else str(genre),
                    'features': features_array
                })
        except Exception as e:
            print(f"Error with {filepath}: {e}")
            continue

    return data


def create_model(input_shape, num_genres):
    """Create CNN model."""
    model = Sequential([
        Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_genres, activation='softmax')
    ])
    return model


def train_from_h5(h5_dir, output_dir='.'):
    """Train model from H5 files."""
    print("=" * 60)
    print("Training from H5 files")
    print("=" * 60)

    # Extract features
    print("\n1. Extracting features from H5 files...")
    data = extract_dataset(h5_dir)
    print(f"   Extracted {len(data)} samples")

    # Convert to arrays
    X = np.array([d['features'] for d in data])
    y = np.array([d['genre'] for d in data])

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   Feature shape: {X.shape}")
    print(f"   Raw feature range: [{X.min():.2f}, {X.max():.2f}]")

    # Get genre counts
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n   Genre distribution:")
    for g, c in zip(unique, counts):
        print(f"      {g}: {c}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_genres = len(label_encoder.classes_)
    print(f"\n   Number of genres: {num_genres}")

    # Scale features using RobustScaler (handles outliers better)
    print("\n2. Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

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

    # Train Random Forest first (often better for tabular data)
    print("\n5. Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"   Random Forest Accuracy: {rf_acc:.2%}")

    # Train CNN
    print("\n6. Training CNN...")
    y_train_oh = to_categorical(y_train, num_genres)
    y_test_oh = to_categorical(y_test, num_genres)

    X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = create_model((k_features, 1), num_genres)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    cw = class_weight.compute_class_weight('balanced', classes=np.arange(num_genres), y=y_train)
    cw_dict = dict(enumerate(cw))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(f'{output_dir}/best_model.keras', save_best_only=True, monitor='val_loss')
    ]

    model.fit(
        X_train_3d, y_train_oh,
        batch_size=32,
        epochs=150,
        validation_data=(X_test_3d, y_test_oh),
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=2
    )

    cnn_loss, cnn_acc = model.evaluate(X_test_3d, y_test_oh, verbose=0)
    print(f"   CNN Accuracy: {cnn_acc:.2%}")

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Random Forest: {rf_acc:.2%}")
    print(f"  CNN: {cnn_acc:.2%}")
    print(f"{'='*60}")

    # Use the better model for classification report
    if rf_acc > cnn_acc:
        y_pred = rf.predict(X_test)
        print("\nRandom Forest Classification Report:")
    else:
        y_pred = np.argmax(model.predict(X_test_3d, verbose=0), axis=1)
        print("\nCNN Classification Report:")

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    print("\nPer-class accuracy:")
    for i, genre in enumerate(label_encoder.classes_):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {genre}: {class_acc:.2%}")

    # Save models and preprocessors
    print("\n7. Saving models...")
    model.save(f'{output_dir}/trained_model.keras')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(selector, f'{output_dir}/selector.pkl')
    joblib.dump(rf, f'{output_dir}/random_forest.pkl')

    print(f"\nSaved to {output_dir}/:")
    print("  - trained_model.keras")
    print("  - best_model.keras")
    print("  - label_encoder.pkl")
    print("  - scaler.pkl")
    print("  - selector.pkl")
    print("  - random_forest.pkl")

    return model, max(rf_acc, cnn_acc)


if __name__ == '__main__':
    train_from_h5('processed_h5', '.')

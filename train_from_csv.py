"""
Quick training script that uses pre-extracted features from CSV.
This allows retraining without the original H5 files.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.optimizers import Adam


def train_from_csv(csv_path, output_dir='.'):
    """Train the model using pre-extracted features from CSV."""

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Separate features and labels
    label_col = 'genre'
    feature_cols = [c for c in df.columns if c not in ['filename', 'genre']]

    X = df[feature_cols].values
    y = df[label_col].values

    # Handle any NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Genres: {np.unique(y)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_genres = len(label_encoder.classes_)
    print(f"Number of genres: {num_genres}")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection - select top features
    k_features = min(100, X_scaled.shape[1])
    selector = SelectKBest(f_classif, k=k_features)
    X_selected = selector.fit_transform(X_scaled, y_encoded)
    print(f"Selected {X_selected.shape[1]} features")

    # Convert labels to one-hot
    y_onehot = to_categorical(y_encoded)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_onehot, test_size=0.2, random_state=42,
        stratify=np.argmax(y_onehot, axis=1)
    )

    # Reshape for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build model
    input_shape = (X_train.shape[1], 1)
    model = Sequential([
        Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(128, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(256, 3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_genres, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.summary()

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.arange(num_genres), y=np.argmax(y_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights))

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy:.2%}")

    # Save everything
    model.save(f'{output_dir}/trained_model.keras')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(selector, f'{output_dir}/selector.pkl')

    print(f"\nSaved to {output_dir}/:")
    print("  - trained_model.keras")
    print("  - label_encoder.pkl")
    print("  - scaler.pkl")
    print("  - selector.pkl")

    return model, history


if __name__ == '__main__':
    train_from_csv('optimized_datasets/processed_dataset_summary.csv', '.')

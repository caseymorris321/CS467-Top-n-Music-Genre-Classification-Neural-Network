"""
Optimized training with ensemble approach combining multiple models.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')


def create_cnn_model(input_shape, num_genres):
    """CNN model optimized for the data."""
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


def train_optimized(csv_path, output_dir='.'):
    """Train with ensemble approach."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Separate features and labels
    exclude_cols = ['filename', 'genre']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    y = df['genre'].values

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_genres = len(label_encoder.classes_)
    print(f"Number of genres: {num_genres}")
    print(f"Classes: {label_encoder.classes_}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use all features for tree-based models (they handle feature selection internally)
    n_features = X_scaled.shape[1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train Random Forest (often works well with tabular data)
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_acc:.2%}")

    # Train Gradient Boosting
    print("\n--- Training Gradient Boosting ---")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_acc = gb.score(X_test, y_test)
    print(f"Gradient Boosting Accuracy: {gb_acc:.2%}")

    # Feature selection for neural network
    k_features = min(50, n_features)
    selector = SelectKBest(f_classif, k=k_features)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # Train CNN
    print("\n--- Training CNN ---")
    y_train_oh = to_categorical(y_train, num_genres)
    y_test_oh = to_categorical(y_test, num_genres)

    X_train_3d = X_train_sel.reshape(X_train_sel.shape[0], X_train_sel.shape[1], 1)
    X_test_3d = X_test_sel.reshape(X_test_sel.shape[0], X_test_sel.shape[1], 1)

    cnn = create_cnn_model((k_features, 1), num_genres)
    cnn.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    cw = class_weight.compute_class_weight('balanced', classes=np.arange(num_genres), y=y_train)
    cw_dict = dict(enumerate(cw))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]

    cnn.fit(
        X_train_3d, y_train_oh,
        batch_size=32,
        epochs=150,
        validation_data=(X_test_3d, y_test_oh),
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=2
    )

    cnn_loss, cnn_acc = cnn.evaluate(X_test_3d, y_test_oh, verbose=0)
    print(f"CNN Accuracy: {cnn_acc:.2%}")

    # Ensemble predictions (weighted average)
    print("\n--- Ensemble Predictions ---")

    # Get predictions from each model
    rf_proba = rf.predict_proba(X_test)
    gb_proba = gb.predict_proba(X_test)
    cnn_proba = cnn.predict(X_test_3d, verbose=0)

    # Weighted average (give more weight to better performing models)
    weights = np.array([rf_acc, gb_acc, cnn_acc])
    weights = weights / weights.sum()
    print(f"Weights: RF={weights[0]:.3f}, GB={weights[1]:.3f}, CNN={weights[2]:.3f}")

    ensemble_proba = weights[0] * rf_proba + weights[1] * gb_proba + weights[2] * cnn_proba
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_acc = np.mean(ensemble_pred == y_test)

    print(f"\n{'='*60}")
    print(f"Random Forest: {rf_acc:.2%}")
    print(f"Gradient Boosting: {gb_acc:.2%}")
    print(f"CNN: {cnn_acc:.2%}")
    print(f"ENSEMBLE ACCURACY: {ensemble_acc:.2%}")
    print(f"{'='*60}")

    # Use best single model or ensemble
    if ensemble_acc >= max(rf_acc, gb_acc, cnn_acc):
        print("Using Ensemble model")
        best_model_name = "ensemble"
    elif rf_acc >= max(gb_acc, cnn_acc):
        print("Using Random Forest model")
        best_model_name = "rf"
    elif gb_acc >= cnn_acc:
        print("Using Gradient Boosting model")
        best_model_name = "gb"
    else:
        print("Using CNN model")
        best_model_name = "cnn"

    # Classification report for ensemble
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=label_encoder.classes_))

    # Per-class accuracy
    cm = confusion_matrix(y_test, ensemble_pred)
    print("\nPer-class accuracy:")
    for i, genre in enumerate(label_encoder.classes_):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  {genre}: {class_acc:.2%}")

    # Save the CNN model (for compatibility with existing inference code)
    cnn.save(f'{output_dir}/trained_model.keras')
    cnn.save(f'{output_dir}/best_model.keras')
    joblib.dump(label_encoder, f'{output_dir}/label_encoder.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    joblib.dump(selector, f'{output_dir}/selector.pkl')

    # Also save RF as backup (often better for tabular data)
    joblib.dump(rf, f'{output_dir}/random_forest.pkl')
    joblib.dump(gb, f'{output_dir}/gradient_boosting.pkl')

    print(f"\nSaved to {output_dir}/")

    return cnn, ensemble_acc


if __name__ == '__main__':
    train_optimized('optimized_datasets/processed_dataset_summary.csv', '.')

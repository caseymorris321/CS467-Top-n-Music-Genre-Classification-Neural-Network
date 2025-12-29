"""
This file contains the primary functions that allow the NN to guess what genres the inputted
song might be related to.
"""

from keras.models import load_model
from optimized_datasets import load_data, gtzan_to_h5
from optimized_datasets.feature_extraction import extract_features
import h5py
import joblib
import numpy as np
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


# Feature order must match training data exactly
FEATURE_ORDER = [
    'duration', 'tempo', 'key', 'mode', 'time_signature', 'rhythmic_complexity',
    'spectral_centroid_mean', 'spectral_rolloff_mean', 'mfcc1', 'mfcc2', 'mfcc3',
    'timbre_std', 'timbre_skew', 'timbre_kurtosis', 'spec_bw', 'zcr', 'rms',
    'timbre_max', 'timbre_min', 'timbre_range', 'timbre_median', 'timbre_variance',
    'timbre_mad', 'timbre_temporal_centroid', 'timbre_q1', 'timbre_q3', 'timbre_iqr',
    'timbre_entropy', 'timbre_energy', 'timbre_flux', 'timbre_flatness',
    'num_non_zero_rms_segments', 'brightness', 'loudness_mean', 'loudness_std',
    'loudness_skew', 'loudness_kurtosis', 'loudness_max', 'loudness_min',
    'loudness_range', 'loudness_median', 'loudness_q1', 'loudness_q3', 'loudness_iqr',
    'loudness_entropy', 'loudness_energy', 'loudness_flux', 'roughness',
    'chroma_mean', 'chroma_std', 'chroma_skew', 'chroma_kurtosis', 'chroma_max',
    'chroma_min', 'chroma_range', 'chroma_median', 'chroma_q1', 'chroma_q3',
    'chroma_iqr', 'chroma_entropy', 'chroma_energy', 'chroma_flux', 'chroma_flatness',
    'melodic_contour_direction', 'melodic_contour_interval', 'rhythmic_entropy',
    'rhythmic_irregularity'
]


def flatten_dict_ordered(d):
    """Flatten a dictionary into a list of values in the correct feature order."""
    result = []
    for key in FEATURE_ORDER:
        if key in d:
            val = d[key]
            if isinstance(val, (int, float, np.integer, np.floating)):
                result.append(float(val))
            else:
                result.append(0.0)
        else:
            result.append(0.0)  # Missing feature gets 0
    return result


def flatten_dict(d):
    """Flatten a dictionary into a single list of values."""
    return flatten_dict_ordered(d)


def interpret_predictions(predictions, label_encoder):
    """
    Interpret model predictions to map them to genre labels.

    :param predictions: Array of model predictions (probability distributions).
    :param label_encoder: LabelEncoder used to decode genre labels.
    :return: Dictionary with genre labels and their corresponding probabilities.
    """
    genre_probabilities = predictions[0]

    # Get the genre labels from the label_encoder
    genre_labels = label_encoder.classes_

    # Create a dictionary to map genre labels to their probabilities
    genre_dict = {genre: prob for genre, prob in zip(genre_labels, genre_probabilities)}

    return genre_dict


def genre_guesser(model_path, file, use_rf=True):
    """
    The genre_guesser guesses what genres are related to the given song data.

    :param model_path: Path to the trained model (or directory containing models)
    :param file: Path to the audio file to classify
    :param use_rf: If True, use Random Forest model (better accuracy)
    :return: A dictionary with the percentile relationship to each genre in the NN
    """
    # Get model directory
    model_dir = os.path.dirname(model_path) or '.'

    # Try to load preprocessors (scaler and selector)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    selector_path = os.path.join(model_dir, 'selector.pkl')
    rf_path = os.path.join(model_dir, 'random_forest.pkl')

    scaler = None
    selector = None
    rf_model = None

    if os.path.exists(scaler_path) and os.path.exists(selector_path):
        scaler = joblib.load(scaler_path)
        selector = joblib.load(selector_path)

    # Also check current directory
    if scaler is None and os.path.exists('scaler.pkl') and os.path.exists('selector.pkl'):
        scaler = joblib.load('scaler.pkl')
        selector = joblib.load('selector.pkl')

    # Try to load Random Forest if available and requested
    if use_rf:
        if os.path.exists(rf_path):
            rf_model = joblib.load(rf_path)
        elif os.path.exists('random_forest.pkl'):
            rf_model = joblib.load('random_forest.pkl')

    # extract features from h5
    file_name = file.rsplit('.', 1)[0]
    h5_file_name = file_name + '.h5'

    gtzan_to_h5.convert_gtzan_to_msd_structure(file, h5_file_name)

    with h5py.File(h5_file_name, 'r') as f:
        try:
            feature = extract_features(f)
            # If feature is a dictionary, flatten it in correct order
            if isinstance(feature, dict):
                feature = flatten_dict(feature)
            feature = np.array(feature, dtype=float).reshape(1, -1)
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

            if scaler is not None and selector is not None:
                # Apply same preprocessing as training
                feature = scaler.transform(feature)
                feature = selector.transform(feature)

            # Use Random Forest if available (better accuracy)
            if rf_model is not None:
                prediction = rf_model.predict_proba(feature)
                return prediction

            # Fall back to neural network
            num_features = feature.shape[1]
            # Reshape for model input (batch_size=1, features, channels=1)
            feature = feature.reshape((1, num_features, 1))

        except Exception as e:
            print(f"Error processing file {h5_file_name}: {str(e)}")
            raise

    # Load and use Keras model
    model = load_model(model_path)
    prediction = model.predict(feature, verbose=0)
    return prediction


if __name__ == '__main__':
    model = "./nn_training/trained_model.keras"
    input_file = 'D:/CS467-Top-n-Music-Genre-Classification-Neural-Network/ui/nirvana_smells_like_teen_spirit.wav'

    results = genre_guesser(model, input_file)

    # Load the LabelEncoder
    label_encoder = joblib.load('./nn_training/label_encoder.pkl')

    print(interpret_predictions(results, label_encoder))

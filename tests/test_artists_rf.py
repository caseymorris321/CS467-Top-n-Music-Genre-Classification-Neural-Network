"""Test the genre classifier using Random Forest on sample artist files."""

import os
import sys
import joblib
import numpy as np
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from optimized_datasets import gtzan_to_h5
from optimized_datasets.feature_extraction import extract_features
from nn_genre_guesser.genre_guesser import FEATURE_ORDER, flatten_dict_ordered


def test_file_rf(file_path, rf_model, scaler, selector, label_encoder):
    """Test a single file using Random Forest."""
    try:
        # Convert to H5
        file_name = file_path.rsplit('.', 1)[0]
        h5_file_name = file_name + '.h5'
        gtzan_to_h5.convert_gtzan_to_msd_structure(file_path, h5_file_name)

        # Extract features
        with h5py.File(h5_file_name, 'r') as f:
            features_dict = extract_features(f)

        # Convert to array in correct order
        features_array = np.array(flatten_dict_ordered(features_dict), dtype=float).reshape(1, -1)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply preprocessing
        features_scaled = scaler.transform(features_array)
        features_selected = selector.transform(features_scaled)

        # Predict probabilities
        proba = rf_model.predict_proba(features_selected)[0]

        # Create results dict
        results = {genre: prob for genre, prob in zip(label_encoder.classes_, proba)}
        return results
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Load models
    rf = joblib.load("random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Test files with expected genres (after genre merging: pop_rock->rock, hiphop->rap)
    test_files = [
        ("ui/nirvana_smells_like_teen_spirit.wav", "rock"),  # pop_rock merged into rock
        ("ui/Carrie_Underwood_-_Jesus_Takes_The_Wheel_(mp3.pm).mp3", "country"),
        ("ui/BLACK SABBATH - Paranoid (Official Video).mp3", "metal"),
        ("ui/Shakira - Hips Don't Lie (featuring Wyclef Jean) (Official 4K Video) ft. Wyclef Jean.mp3", "latin"),
        ("ui/The Notorious B.I.G. - Juicy (Official Video) [4K].mp3", "rap"),  # hiphop merged into rap
    ]

    print("=" * 70)
    print("Genre Classification Test Results (Random Forest)")
    print("=" * 70)

    correct = 0
    total = 0

    for file_path, expected_genre in test_files:
        if not os.path.exists(file_path):
            print(f"\nFile not found: {file_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Testing: {os.path.basename(file_path)}")
        print(f"Expected genre: {expected_genre}")
        print("-" * 70)

        results = test_file_rf(file_path, rf, scaler, selector, label_encoder)

        if results:
            # Sort by confidence
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

            # Top 5 predictions
            print("Top 5 predictions:")
            for i, (genre, confidence) in enumerate(sorted_results[:5]):
                marker = " <-- EXPECTED" if genre == expected_genre else ""
                print(f"  {i+1}. {genre}: {confidence:.2%}{marker}")

            # Check if expected genre is in top prediction
            predicted_genre = sorted_results[0][0]
            if predicted_genre == expected_genre:
                print("\n[CORRECT]")
                correct += 1
            else:
                # Check if expected is in top 5
                top5_genres = [g for g, _ in sorted_results[:5]]
                if expected_genre in top5_genres:
                    rank = top5_genres.index(expected_genre) + 1
                    print(f"\n[CLOSE] Expected genre ranked #{rank}")
                else:
                    print(f"\n[MISS] Expected genre not in top 5")

            total += 1

    print("\n" + "=" * 70)
    print(f"Final Score: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()

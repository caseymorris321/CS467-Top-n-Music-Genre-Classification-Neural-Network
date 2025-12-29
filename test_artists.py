"""Test the genre classifier on sample artist files."""

import os
import sys
import joblib
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nn_genre_guesser.genre_guesser import genre_guesser, interpret_predictions

def test_file(file_path, model_path, label_encoder):
    """Test a single file and return results."""
    try:
        prediction = genre_guesser(model_path, file_path)
        results = interpret_predictions(prediction, label_encoder)
        return results
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    model_path = "trained_model.keras"
    label_encoder = joblib.load("label_encoder.pkl")

    # Test files with expected genres (genre merging applied: pop_rock->rock, hiphop->rap)
    test_files = [
        ("ui/nirvana_smells_like_teen_spirit.wav", "rock"),  # pop_rock merged into rock
        ("ui/Carrie_Underwood_-_Jesus_Takes_The_Wheel_(mp3.pm).mp3", "country"),
        ("ui/BLACK SABBATH - Paranoid (Official Video).mp3", "metal"),
        ("ui/Shakira - Hips Don't Lie (featuring Wyclef Jean) (Official 4K Video) ft. Wyclef Jean.mp3", "latin"),
        ("ui/The Notorious B.I.G. - Juicy (Official Video) [4K].mp3", "rap"),  # hiphop merged into rap
    ]

    print("=" * 70)
    print("Genre Classification Test Results")
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

        results = test_file(file_path, model_path, label_encoder)

        if results:
            # Sort by confidence
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

            # Top 3 predictions
            print("Top 3 predictions:")
            for i, (genre, confidence) in enumerate(sorted_results[:3]):
                marker = " <-- EXPECTED" if genre == expected_genre else ""
                print(f"  {i+1}. {genre}: {confidence:.2%}{marker}")

            # Check if expected genre is in top prediction
            predicted_genre = sorted_results[0][0]
            if predicted_genre == expected_genre:
                print("\n[CORRECT]")
                correct += 1
            else:
                # Check if expected is in top 3
                top3_genres = [g for g, _ in sorted_results[:3]]
                if expected_genre in top3_genres:
                    rank = top3_genres.index(expected_genre) + 1
                    print(f"\n[CLOSE] Expected genre ranked #{rank}")
                else:
                    print(f"\n[MISS] Expected genre not in top 3")

            total += 1

    print("\n" + "=" * 70)
    print(f"Final Score: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()

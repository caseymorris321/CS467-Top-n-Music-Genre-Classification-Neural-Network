import sys
import time
import threading
import os
import joblib
import numpy as np

from typing import Dict
from nn_genre_guesser.genre_guesser import genre_guesser, interpret_predictions
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def classify_genre(file_path: str, model_path: str, label_encoder_path: str) -> Dict[str, float]:
    # Use the genre_guesser function to get the features
        prediction = genre_guesser(model_path, file_path)
        return prediction


def animate_processing():
    chars = "\\-/|"
    i = 0
    while getattr(threading.current_thread(), "do_run", True):
        sys.stdout.write(f"\rProcessing... {chars[i % len(chars)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

def get_file_path() -> str:
    while True:
        file_path = input("Please input the file you would like to have tested: ")
        if os.path.isfile(file_path):
            return file_path
        else:
            print("Error: File not found. Please enter a valid file path.")


def display_results(results: Dict[str, float]):
    print("\nResults:")
    for genre, confidence in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{genre}: {confidence:.2%}")

def get_user_choice() -> bool:
    while True:
        choice = input("\nClassify another song? (Y to restart / N to quit): ").lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

def main():
    model_path = "trained_model.keras"
    label_encoder_path = "label_encoder.pkl"

    while True:
        print("\nWelcome to Top-n Music Genre Classification Neural Network")

        file_path = get_file_path()

        animation_thread = threading.Thread(target=animate_processing)
        animation_thread.daemon = True
        animation_thread.start()

        try:
            results = classify_genre(file_path, model_path, label_encoder_path)
            animation_thread.do_run = False
            animation_thread.join()
            print("\rAnalysis complete!")
            time.sleep(0.5)
            display_results(results)
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
        finally:
            animation_thread.do_run = False
            animation_thread.join()

        if not get_user_choice():
            break

    print("\nThank you for using the Top-n Music Genre Classification Neural Network. Goodbye!")

if __name__ == "__main__":
    main()

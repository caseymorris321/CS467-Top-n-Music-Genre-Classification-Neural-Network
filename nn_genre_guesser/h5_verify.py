import h5py
import librosa
import numpy as np


def print_h5_structure(file_path):
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    with h5py.File(file_path, 'r') as h5file:
        print(h5file.attrs['genre'])
        h5file.visititems(print_structure)


def extract_features_labels_complex_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5file:
        features = []
        labels = []

        # Check if the file contains the necessary groups and datasets
        if 'analysis' in h5file and 'metadata' in h5file:
            analysis_group = h5file['analysis']
            metadata_group = h5file['metadata']

            # Extract relevant features from the analysis group
            bars_confidence = np.array(analysis_group['bars_confidence'])
            beats_confidence = np.array(analysis_group['beats_confidence'])
            sections_confidence = np.array(analysis_group['sections_confidence'])
            segments_loudness_max = np.array(analysis_group['segments_loudness_max'])
            segments_pitches = np.array(analysis_group['segments_pitches'])
            segments_timbre = np.array(analysis_group['segments_timbre'])

            # Flatten and concatenate features into a single array per song
            song_features = np.concatenate((
                bars_confidence.flatten(),
                beats_confidence.flatten(),
                sections_confidence.flatten(),
                segments_loudness_max.flatten(),
                segments_pitches.flatten(),
                segments_timbre.flatten()
            ), axis=None)
            features.append(song_features)

            # Extract the genre label from the metadata group
            metadata_songs = metadata_group['songs']
            genre = metadata_songs['genre'][0].decode('utf-8')
            labels.append(genre)

        return np.array(features), np.array(labels)


def extract_mfcc_features_from_simple_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5file:
        if 'audio' in h5file and 'sr' in h5file:
            audio = np.array(h5file['audio'])
            sample_rate = int(h5file['sr'][()])

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            return mfccs_scaled


# Example usage
h5_file_path = '../optimized_datasets/processed_h5/blues.00000.h5'

print_h5_structure(h5_file_path)
# print(extract_mfcc_features_from_simple_h5(h5_file_path))

arr1, arr2 = extract_features_labels_complex_h5(h5_file_path)
print(arr1, arr2)

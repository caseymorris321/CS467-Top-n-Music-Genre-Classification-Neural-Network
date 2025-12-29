import h5py
import os
import librosa
import numpy as np


def analyze_msd_structure(msd_file):
    """
    Analyze the structure of a Million Song Dataset (MSD) file.

    :param msd_file: Path to the MSD file
    :return: Dictionary containing the structure of the file
    """
    with h5py.File(msd_file, 'r') as f:
        structure = {}

        def visit_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                structure[name] = obj.shape
        f.visititems(visit_item)
    return structure


def convert_gtzan_to_msd_structure(gtzan_file, output_file):
    """
    Convert a GTZAN audio file to MSD structure and save as H5 file.

    :param gtzan_file: Path to the input GTZAN audio file
    :param output_file: Path to save the output H5 file
    """
    # Load audio file
    audio, sr = librosa.load(gtzan_file, sr=None)

    with h5py.File(output_file, 'w') as f:
        # Extract audio features
        duration = librosa.get_duration(y=audio, sr=sr)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[
            0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        loudness = librosa.feature.rms(y=audio)[0]
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        # Determine key and mode
        chroma_sum = np.sum(chroma, axis=1)
        key = np.argmax(chroma_sum)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                                  5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                                  4.75, 3.98, 2.69, 3.34, 3.17])
        major_correlation = np.correlate(
            chroma_sum, np.roll(major_profile, key))
        minor_correlation = np.correlate(
            chroma_sum, np.roll(minor_profile, key))
        mode = 0 if major_correlation > minor_correlation else 1

        # Determine time signature
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        time_signature = 4 if len(beats) % 3 != 0 else 3

        # Create analysis group and datasets
        analysis = f.create_group('analysis')
        songs = analysis.create_dataset('songs', (1,), dtype=[
            ('duration', '<f8'), ('tempo', '<f8'), ('key', '<i4'),
            ('mode', '<i4'), ('time_signature', '<i4'),
            ('spectral_centroid', '<f8'), ('spectral_rolloff', '<f8')
        ])
        songs[0] = (duration, tempo, key, mode, time_signature,
                    np.mean(spectral_centroids), np.mean(spectral_rolloff))

        # Create additional datasets
        analysis.create_dataset('segments_timbre', data=mfccs.T)
        analysis.create_dataset('segments_pitches', data=chroma.T)
        analysis.create_dataset('segments_loudness_max', data=loudness)
        analysis.create_dataset('segments_zero_crossing_rate',
                                data=zero_crossing_rate)
        analysis.create_dataset('segments_spectral_contrast',
                                data=spectral_contrast.T)

        # Set genre attribute
        f.attrs['genre'] = os.path.basename(gtzan_file).split('.')[0]


def main():
    """
    Main function to analyze MSD structure and convert GTZAN files to MSD structure.
    """
    # Analyze MSD structure
    msd_file = ('C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network/'
                'Github-ignore/unified dataset/msd_output/TRAAAAW128F429D538.h5')
    analyze_msd_structure(msd_file)

    # Convert GTZAN files to MSD structure
    gtzan_dir = ('C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network/'
                 'Github-ignore/unified dataset/gtzan-input')
    output_dir = ('C:/Users/wwwhu/PycharmProjects/CS467-Top-n-Music-Genre-Classification-Neural-Network/'
                  'Github-ignore/unified dataset/gtzan-output')
    os.makedirs(output_dir, exist_ok=True)

    for genre in os.listdir(gtzan_dir):
        genre_path = os.path.join(gtzan_dir, genre)
        print(f'Converting song in genre: {genre}')
        if os.path.isdir(genre_path):
            for track in os.listdir(genre_path):
                if track.endswith('.au'):
                    input_file = os.path.join(genre_path, track)
                    output_file = os.path.join(output_dir, f"{track[:-3]}.h5")
                    convert_gtzan_to_msd_structure(input_file, output_file)

    print("GTZAN conversion to MSD structure complete.")


if __name__ == "__main__":
    main()

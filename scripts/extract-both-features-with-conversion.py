import os
import numpy as np
import pandas as pd
import librosa
import h5py

def extract_gtzan_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    
    duration = librosa.get_duration(y=audio, sr=sr)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    loudness = librosa.feature.rms(y=audio)[0]
    
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    
    chroma_sum = np.sum(chroma, axis=1)
    key = np.argmax(chroma_sum)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_correlation = np.correlate(chroma_sum, np.roll(major_profile, key))
    minor_correlation = np.correlate(chroma_sum, np.roll(minor_profile, key))
    mode = 0 if major_correlation > minor_correlation else 1  # 0 for major, 1 for minor

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    time_signature = 4 if len(beats) % 3 != 0 else 3

    return {
        'duration': duration,
        'spectral_centroid_mean': np.mean(spectral_centroids),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'tempo': tempo,
        'mfcc1': np.mean(mfccs[0]),
        'mfcc2': np.mean(mfccs[1]),
        'mfcc3': np.mean(mfccs[2]),
        'loudness_mean': np.mean(loudness),
        'key': key,
        'mode': mode,
        'time_signature': time_signature,
        'chroma_mean': np.mean(chroma, axis=1).tolist()
    }

def extract_msd_features(file_path):
    with h5py.File(file_path, 'r') as f:
        duration = f['analysis/songs']['duration'][0] if 'duration' in f['analysis/songs'].dtype.names else None
        segments_timbre = f['analysis/segments_timbre'][:] if 'analysis/segments_timbre' in f else None
        segments_pitches = f['analysis/segments_pitches'][:] if 'analysis/segments_pitches' in f else None
        tempo = f['analysis/songs']['tempo'][0] if 'tempo' in f['analysis/songs'].dtype.names else None
        segments_loudness_max = f['analysis/segments_loudness_max'][:] if 'analysis/segments_loudness_max' in f else None
        key = f['analysis/songs']['key'][0] if 'key' in f['analysis/songs'].dtype.names else None
        mode = f['analysis/songs']['mode'][0] if 'mode' in f['analysis/songs'].dtype.names else None
        time_signature = f['analysis/songs']['time_signature'][0] if 'time_signature' in f['analysis/songs'].dtype.names else None

    return {
        'duration': duration,
        'spectral_centroid_mean': np.mean(segments_timbre) if segments_timbre is not None else None,
        'spectral_rolloff_mean': np.mean(segments_pitches) if segments_pitches is not None else None,
        'tempo': tempo,
        'mfcc1': np.mean(segments_timbre[:, 0]) if segments_timbre is not None else None,
        'mfcc2': np.mean(segments_timbre[:, 1]) if segments_timbre is not None else None,
        'mfcc3': np.mean(segments_timbre[:, 2]) if segments_timbre is not None else None,
        'loudness_mean': np.mean(segments_loudness_max) if segments_loudness_max is not None else None,
        'key': key,
        'mode': mode,
        'time_signature': time_signature,
        'chroma_mean': np.mean(segments_pitches, axis=0).tolist() if segments_pitches is not None else None
    }

# New function to create GTZAN H5 files with features
def create_gtzan_h5_with_features(audio_file, output_file):
    audio, sr = librosa.load(audio_file, sr=None)
    features = extract_gtzan_features(audio_file)
    
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('audio', data=audio)
        hf.create_dataset('sr', data=sr)
        for key, value in features.items():
            hf.create_dataset(f'features/{key}', data=value)

# Extract GTZAN features and create H5 files
gtzan_data = []
os.makedirs('./gtzan_h5', exist_ok=True)  # Create directory for GTZAN H5 files
for genre in os.listdir('./genres'):
    genre_path = os.path.join('./genres', genre)
    if os.path.isdir(genre_path):
        for track in os.listdir(genre_path):
            if track.endswith('.au'):
                track_path = os.path.join(genre_path, track)
                output_file = os.path.join('./gtzan_h5', f"{genre}_{track[:-3]}.h5")
                create_gtzan_h5_with_features(track_path, output_file)
                
                features = extract_gtzan_features(track_path)
                features['genre'] = genre
                features['filename'] = f"{genre}_{track[:-3]}.h5"  # Update filename to H5
                gtzan_data.append(features)

gtzan_df = pd.DataFrame(gtzan_data)
gtzan_df.to_csv('gtzan_features.csv', index=False)

# Load unified dataset
unified_df = pd.read_csv('unified_music_dataset.csv')

# Extract MSD features only for files in unified dataset
msd_data = []
for _, row in unified_df.iterrows():
    if row['filename'].endswith('.h5'):
        file_path = os.path.join('./millionsongsubset', row['filename'])
        if os.path.exists(file_path):
            features = extract_msd_features(file_path)
            features['filename'] = row['filename']
            features['genre'] = row['genre']
            msd_data.append(features)

msd_df = pd.DataFrame(msd_data)
msd_df.to_csv('msd_features.csv', index=False)

print("Feature extraction and H5 conversion complete. Features saved to CSV files and GTZAN converted to H5.")
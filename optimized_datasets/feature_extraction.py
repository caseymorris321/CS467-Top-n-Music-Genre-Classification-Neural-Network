import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, entropy, gmean
from librosa.feature import spectral_bandwidth, zero_crossing_rate, rms


def safe_get(f, key, default_value):
    """Safely retrieve a value from an H5 file, returning a default if not found."""
    try:
        value = f[key]
        if isinstance(value, h5py.Dataset):
            if value.shape == ():  # scalar dataset
                return value[()]
            else:
                return value[:]  # return as numpy array
        else:
            return value
    except (KeyError, ValueError):
        return default_value


def extract_features(f):
    """Extract various audio features from an H5 file."""
    features = {}

    # Extract song-level features
    songs = safe_get(f, 'analysis/songs', None)
    if songs is not None:
        if isinstance(songs, np.ndarray) and songs.dtype.names:
            features['duration'] = songs['duration'][0] if 'duration' in songs.dtype.names else 0
            features['tempo'] = songs['tempo'][0] if 'tempo' in songs.dtype.names else 0
            features['key'] = int(
                songs['key'][0]) if 'key' in songs.dtype.names else 0
            features['mode'] = int(
                songs['mode'][0]) if 'mode' in songs.dtype.names else 0
            features['time_signature'] = int(
                songs['time_signature'][0]) if 'time_signature' in songs.dtype.names else 4
        else:
            features['duration'] = safe_get(f, 'analysis/songs/duration', 0)
            features['tempo'] = safe_get(f, 'analysis/songs/tempo', 0)
            features['key'] = int(safe_get(f, 'analysis/songs/key', 0))
            features['mode'] = int(safe_get(f, 'analysis/songs/mode', 0))
            features['time_signature'] = int(
                safe_get(f, 'analysis/songs/time_signature', 4))

        features['rhythmic_complexity'] = features['tempo'] * \
            features['time_signature'] / 120

    # Extract timbre-related features
    segments_timbre = safe_get(f, 'analysis/segments_timbre', None)
    if segments_timbre is not None:
        timbre = np.array(segments_timbre)
        if timbre.ndim == 2 and timbre.shape[0] > 0:
            timbre_normalized = timbre / \
                (np.sum(np.abs(timbre), axis=1, keepdims=True) + 1e-10)
            features['spectral_centroid_mean'] = np.nanmean(timbre[:, 0])
            features['spectral_rolloff_mean'] = np.nanmean(timbre[:, 1])
            features['mfcc1'] = np.nanmean(timbre[:, 0])
            features['mfcc2'] = np.nanmean(timbre[:, 1])
            features['mfcc3'] = np.nanmean(timbre[:, 2])
            features['timbre_std'] = np.nanmean(np.nanstd(timbre, axis=0))
            features['timbre_skew'] = np.nanmean(
                skew(timbre, axis=0, nan_policy='omit'))
            features['timbre_kurtosis'] = np.nanmean(
                kurtosis(timbre, axis=0, nan_policy='omit'))

            n_fft = min(2048, timbre.shape[0])
            features['spec_bw'] = np.nanmean(
                spectral_bandwidth(y=timbre.T, sr=22050, n_fft=n_fft)[0])
            features['zcr'] = np.nanmean(zero_crossing_rate(y=timbre.T)[0])
            features['rms'] = np.nanmean(rms(y=timbre.T)[0])

            features['timbre_max'] = np.nanmax(timbre, axis=0).mean()
            features['timbre_min'] = np.nanmin(timbre, axis=0).mean()
            features['timbre_range'] = features['timbre_max'] - \
                features['timbre_min']
            features['timbre_median'] = np.nanmedian(timbre, axis=0).mean()
            features['timbre_variance'] = np.nanmean(np.nanvar(timbre, axis=0))
            features['timbre_mad'] = np.nanmean(
                np.abs(np.diff(timbre, axis=0)))
            timbre_sum = np.nansum(np.nanmean(timbre, axis=1))
            features['timbre_temporal_centroid'] = np.nanmean(
                np.arange(len(timbre)) * np.nanmean(timbre, axis=1)) / (timbre_sum + 1e-10)

            features['timbre_q1'] = np.nanpercentile(timbre, 25, axis=0).mean()
            features['timbre_q3'] = np.nanpercentile(timbre, 75, axis=0).mean()
            features['timbre_iqr'] = features['timbre_q3'] - \
                features['timbre_q1']
            features['timbre_entropy'] = entropy(
                np.abs(timbre_normalized) + 1e-10, axis=0).mean()
            features['timbre_energy'] = np.nansum(timbre**2, axis=0).mean()
            features['timbre_flux'] = np.nanmean(np.diff(timbre, axis=0)**2)
            features['timbre_flatness'] = gmean(np.abs(
                timbre) + 1e-10, axis=0).mean() / (np.nanmean(np.abs(timbre), axis=0).mean() + 1e-10)

            rms_values = rms(y=timbre.T)[0]
            non_zero_rms_segments = np.sum(rms_values > 1e-10)
            features['num_non_zero_rms_segments'] = non_zero_rms_segments

            features['brightness'] = np.nanmean(
                np.sum(timbre[:, 1:], axis=1) / (np.sum(timbre, axis=1) + 1e-10))

    # Extract loudness-related features
    segments_loudness = safe_get(f, 'analysis/segments_loudness_max', None)
    if segments_loudness is not None:
        loudness = np.array(segments_loudness)
        features['loudness_mean'] = np.nanmean(loudness)
        features['loudness_std'] = np.nanstd(loudness)
        features['loudness_skew'] = skew(loudness, nan_policy='omit')
        features['loudness_kurtosis'] = kurtosis(loudness, nan_policy='omit')
        features['loudness_max'] = np.nanmax(loudness)
        features['loudness_min'] = np.nanmin(loudness)
        features['loudness_range'] = features['loudness_max'] - \
            features['loudness_min']
        features['loudness_median'] = np.nanmedian(loudness)

        features['loudness_q1'] = np.nanpercentile(loudness, 25)
        features['loudness_q3'] = np.nanpercentile(loudness, 75)
        features['loudness_iqr'] = features['loudness_q3'] - \
            features['loudness_q1']
        features['loudness_entropy'] = entropy(np.abs(loudness) + 1e-10)
        features['loudness_energy'] = np.nansum(loudness**2)
        features['loudness_flux'] = np.nanmean(np.diff(loudness)**2)

        features['roughness'] = np.nanmean(np.abs(np.diff(loudness)))

    # Extract loudness-related features
    segments_pitches = safe_get(f, 'analysis/segments_pitches', None)
    if segments_pitches is not None:
        pitches = np.array(segments_pitches)
        pitches = np.maximum(pitches, 0)
        features['chroma_mean'] = np.nanmean(pitches)
        features['chroma_std'] = np.nanstd(pitches)
        features['chroma_skew'] = np.nanmean(
            skew(pitches, axis=0, nan_policy='omit'))
        features['chroma_kurtosis'] = np.nanmean(
            kurtosis(pitches, axis=0, nan_policy='omit'))
        features['chroma_max'] = np.max(np.mean(pitches, axis=0))
        features['chroma_min'] = np.min(np.mean(pitches, axis=0))
        features['chroma_range'] = features['chroma_max'] - \
            features['chroma_min']
        features['chroma_median'] = np.nanmedian(pitches, axis=0).mean()

        pitch_diff = np.diff(pitches, axis=0)
        features['zcr'] = np.nanmean(
            np.sum(np.abs(np.sign(pitch_diff)), axis=1) / (2 * pitches.shape[1]))

        features['chroma_q1'] = np.nanpercentile(pitches, 25, axis=0).mean()
        features['chroma_q3'] = np.nanpercentile(pitches, 75, axis=0).mean()
        features['chroma_iqr'] = features['chroma_q3'] - features['chroma_q1']
        features['chroma_entropy'] = entropy(pitches + 1e-10, axis=0).mean()
        features['chroma_energy'] = np.nansum(pitches**2, axis=0).mean()
        features['chroma_flux'] = np.nanmean(np.diff(pitches, axis=0)**2)
        features['chroma_flatness'] = gmean(
            pitches + 1e-10, axis=0).mean() / (np.nanmean(pitches + 1e-10, axis=0).mean())

        melodic_contour = np.argmax(pitches, axis=1)
        features['melodic_contour_direction'] = np.nanmean(
            np.diff(melodic_contour))
        features['melodic_contour_interval'] = np.nanmean(
            np.abs(np.diff(melodic_contour)))

        # Rhythmic complexity features
        onset_env = np.sum(np.diff(pitches, axis=0) > 0, axis=1)
        peaks, _ = find_peaks(onset_env)
        if len(peaks) > 1:
            ioi = np.diff(peaks)
            features['rhythmic_entropy'] = entropy(ioi)
            features['rhythmic_irregularity'] = np.std(ioi) / np.mean(ioi)

    return features

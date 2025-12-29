import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from optimized_datasets.feature_extraction import extract_features


def find_h5_files(directory):
    """Find all H5 files in the directory and its subdirectories."""
    return [os.path.join(root, file) for root, _, files in os.walk(directory)
            for file in files if file.endswith('.h5')]


def clean_filename(filename):
    """Clean the filename by removing path and whitespace."""
    return os.path.basename(filename).strip()


def process_and_copy_h5(input_files, output_dir, unified_df=None):
    """Process H5 files, extract features, and copy to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    data = []

    files_to_process = input_files if unified_df is None else unified_df['filename']

    for filename in tqdm(files_to_process, desc="Processing files"):
        clean_name = clean_filename(filename)
        input_file = next(
            (f for f in input_files if clean_filename(f) == clean_name), None)
        if input_file:
            output_file = os.path.join(output_dir, clean_name)
            shutil.copy2(input_file, output_file)

            # Update genre attribute
            with h5py.File(output_file, 'r+') as f:
                genre = unified_df.loc[unified_df['filename'] == filename,
                                       'genre'].iloc[0] if unified_df is not None else f.attrs.get('genre')
                f.attrs['genre'] = genre

            # Extract features
            with h5py.File(output_file, 'r') as f:
                summary = {
                    'filename': clean_name,
                    'genre': f.attrs['genre'],
                }

                features = extract_features(f)
                summary.update(features)

            data.append(summary)
        else:
            print(f"File not found: {filename}")

    return pd.DataFrame(data)


def balance_dataset(df, n_samples=300):
    """This function maintains up to 300 samples per genre, including all samples
    for genres with fewer than 300 tracks."""
    return df.groupby('genre').apply(lambda x: x.sample(n=min(len(x), n_samples), random_state=42)).reset_index(drop=True)


if __name__ == "__main__":
    # Load unified dataset to match filenames with millionsongsubset
    unified_df = pd.read_csv('unified_preprocessed_dataset.csv')

    # Clean filenames and genres
    unified_df['filename'] = unified_df['filename'].apply(clean_filename)
    unified_df['genre'] = unified_df['genre'].str.lower()

    # Filter genres with at least 70 tracks
    genre_counts = unified_df['genre'].value_counts()
    genres_to_keep = genre_counts[genre_counts >= 101].index
    unified_df = unified_df[unified_df['genre'].isin(genres_to_keep)]

    # Find all H5 files in the specified directories
    gtzan_files = find_h5_files('./gtzan_msd_structure')
    msd_files = find_h5_files('./millionsongsubset')
    all_files = gtzan_files + msd_files

    # Process H5 files and copy to output directory
    combined_summary = process_and_copy_h5(
        all_files, './processed_h5', unified_df)

    # Balance the dataset
    balanced_summary = balance_dataset(combined_summary)

    # Preprocess numeric columns
    numeric_columns = balanced_summary.select_dtypes(
        include=[np.number]).columns
    non_numeric_columns = balanced_summary.select_dtypes(
        exclude=[np.number]).columns

    balanced_summary[numeric_columns] = balanced_summary[numeric_columns].abs()
    balanced_summary[numeric_columns] = balanced_summary[numeric_columns].replace([
                                                                                  np.inf, -np.inf], np.nan)
    balanced_summary[numeric_columns] = balanced_summary[numeric_columns].fillna(
        balanced_summary[numeric_columns].median())

    # Define feature columns
    feature_columns = [col for col in balanced_summary.columns if col not in [
        'filename', 'genre', 'duration', 'key', 'mode', 'time_signature']]

    # Clip extreme values
    for column in feature_columns:
        if column in numeric_columns:
            balanced_summary[column] = balanced_summary[column].clip(
                -1e15, 1e15)

    # Scale features
    scaler = RobustScaler()
    balanced_summary[feature_columns] = scaler.fit_transform(
        balanced_summary[feature_columns])

    # Prepare features and labels
    X = balanced_summary[['duration', 'rhythmic_complexity'] +
                         feature_columns + ['key', 'mode', 'time_signature']]

    print(f"Number of features: {X.shape[1]}")

    y = balanced_summary['genre']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Save processed data
    balanced_summary.to_csv('processed_dataset_summary.csv', index=False)
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    print("Processing complete. H5 files copied to './processed_h5' and summary saved to 'processed_dataset_summary.csv'.")
    print(
        f"Number of genres in the final dataset: {balanced_summary['genre'].nunique()}")
    print(f"Genre distribution:\n{balanced_summary['genre'].value_counts()}")

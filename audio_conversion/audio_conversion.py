from pydub import AudioSegment
import os
import tkinter as tk
from tkinter import filedialog
import h5py
import numpy as np
from optimized_datasets import feature_extraction


def convert_audio(input_file, output_format):
    """
    Function that will check if the file exists, get its extension,
    load it, get the new file format, and export it
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File {input_file} does not exist.")

    file_name, _ = os.path.splitext(input_file)

    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        raise ValueError(f"Could not load audio file {input_file}: {e}")

    if output_format == 'h5':
        output_file = f"{file_name}.{output_format}"
        try:
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate
            channels = audio.channels

            with h5py.File(output_file, 'w') as f:
                f.create_dataset('audio_data', data=samples)
                f.attrs['sample_rate'] = sample_rate
                f.attrs['channels'] = channels

            print(f"File has been converted to {output_file}")
        except Exception as e:
            raise ValueError(
                f"Could not export audio file to {output_format}: {e}")
    else:
        output_file = f"{file_name}.{output_format}"
        try:
            with open(output_file, 'wb') as f:
                audio.export(f, format=output_format)
            print(f"File has been converted to {output_file}")
        except Exception as e:
            raise ValueError(
                f"Could not export audio file to {output_format}: {e}")
        features = feature_extraction.extract_features(output_file)
        print(f"Extracted features: {features}")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    input_file = filedialog.askopenfilename(title="Select input audio file")
    if not input_file:
        print("No file selected.")
    else:
        supported_formats = ['wav', 'mp3', 'au', 'ogg', 'flac', 'h5']
        output_format = input(
            "Enter desired output format "
            "(example: wav, mp3, au, ogg, flac, h5):"
        ).strip().lower()

        if output_format not in supported_formats:
            supported = ', '.join(supported_formats)
            print(
                f"Unsupported format: {output_format}. "
                f"Supported formats are: {supported}"
            )
        else:
            convert_audio(input_file, output_format)

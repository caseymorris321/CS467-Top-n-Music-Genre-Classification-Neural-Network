from pydub import AudioSegment
import os
import tkinter as tk
from tkinter import filedialog


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

    output_file = f"{file_name}.{output_format}"

    try:
        with open(output_file, 'wb') as f:
            audio.export(f, format=output_format)
        print(f"File has been converted to {output_file}")
    except Exception as e:
        raise ValueError(
            f"Could not export audio file to {output_format}: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    input_file = filedialog.askopenfilename(
        title="Select input audio file")
    if not input_file:
        print("No file selected.")
    else:
        supported_formats = ['wav', 'mp3', 'au', 'ogg', 'flac']
        output_format = input(
            "Enter desired output format (example: wav, mp3, au, ogg, flac):"
            ).strip().lower()

    if output_format not in supported_formats:
        supported = ', '.join(supported_formats)
        print(f"Unsupported format: {output_format}. "
              f"Supported formats are: {supported}")
    else:
        convert_audio(input_file, output_format)

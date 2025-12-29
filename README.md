# CS467-Top-n-Music-Genre-Classification-Neural-Network

## General Overview

A machine learning system that classifies music into 14 genres using audio feature extraction. Uses an ensemble of Random Forest, XGBoost, and Gradient Boosting classifiers trained on 67 audio features extracted via Librosa.

## Installation

### Install the Python Packages

To install the python packages, use the following command once you have copied the repo:

```
pip install -r requirements.txt
```

### Installation Guide for FFmpeg

To run the audio conversion successfully, you need to have FFmpeg installed on your system. Below are the instructions for installing FFmpeg.

1. Download the latest FFmpeg release from the official website: [FFmpeg Downloads](https://ffmpeg.org/download.html)
2. Extract the downloaded ZIP file to a folder on your computer.
3. Add FFmpeg to your system PATH:
   - Open the Start Menu, search for "Environment Variables" and open it.
   - Under the "System variables" section, find the Path variable and select "Edit".
   - Click "New" and add the path to the bin folder inside the extracted FFmpeg directory (e.g., C:\ffmpeg\bin).
   - Click "OK" to close all windows.
4. Open a command prompt and type `ffmpeg -version` to verify the installation.

## Quick Start

To classify a song:

```
python genre_classifier_cli.py path/to/song.mp3
```

To retrain the models:

```
python train_xgboost.py
```

## Neural Network Datasets

The h5 dataset folder contains 1000 audio files in HDF5 format, each tagged with genre metadata. These are used to train the neural network and help identify what genres are being considered.

## Prediction Program

The predict folder contains the files used to process input audio files and receive genre predictions. The main function handles user input, processes audio files, and displays genre predictions using the trained model.

## Audio Conversion

The audio conversion folder contains files to convert audio from one supported format to another. It uses Pydub/FFmpeg libraries. The script takes an input file and desired output format, verifies the file exists, then loads and converts the audio content. For HDF5 output, it stores audio data, sample rate, and channel information. For other formats, it exports directly.

### tests.py

The tests.py file contains unittests including creating a 20-second audio clip in mp3 converted to wav. It tests for successful audio conversion, file existence, correct file format, and supported format handling.

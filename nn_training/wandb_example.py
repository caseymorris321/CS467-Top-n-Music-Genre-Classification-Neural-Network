"""
This file contains the contents necessary to train the neural network

Currently, this file contains code from the following source:
https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy

This is being used to help me better understand how these packages work together. Given our dataset, we will likely
need to take a slightly different route that this
"""


import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import wandb


wandb.init(project='audio_classification_using_W&B', entity='Music classification test',
           config={"batch_size": 32, "epochs": 100})


def load_data(data_path, metadata_path):
    features = []
    labels = []

    metadata = pd.read_csv(metadata_path)

    for index, row in metadata.iterrows():
        file_path = os.path.join(data_path, f"fold{row['fold']}", f"{row['slice_file_name']}")

        # Load the audio file and resample it
        target_sr = 22050
        audio, sample_rate = librosa.load(file_path, sr=target_sr)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        # Append features and labels
        features.append(mfccs_scaled)
        labels.append(row['class'])

    return np.array(features), np.array(labels)


data_path = "/kaggle/input/urbansound8k"
metadata_path = "/kaggle/input/urbansound8k/UrbanSound8K.csv"
features, labels = load_data(data_path, metadata_path)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Split the data into training and testing sets. X data will be used to contain the features
# while y contains the labels.
X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2,
                                                    random_state=42, stratify=labels_onehot)

# Generation of the model. It is a sequential layer model, meaning that each layer will have 1 input and 1 output
input_shape = (X_train.shape[1], 1)
model = Sequential()
model.add(Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# TESTING Portion
def make_predictions(model, le, file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return le.inverse_transform(predicted_class_index)[0]


wandb_table = wandb.Table(columns=["File", "True Label", "Old Prediction", "New Prediction"])

# Save the initial weights
initial_weights = model.get_weights()

# Dictionary to store old predictions
old_predictions = {}

# List of test files and their true labels
test_files = [
    ("/kaggle/input/urbansound8k/fold1/101415-3-0-2.wav", "Dog bark"),
    ("/kaggle/input/urbansound8k/fold1/101415-3-0-3.wav", "Dog bark"),
    ("/kaggle/input/urbansound8k/fold1/102305-6-0-0.wav", "Gun shots"),
    ("/kaggle/input/urbansound8k/fold1/103074-7-0-2.wav", "Jack hammer"),
    ("/kaggle/input/urbansound8k/fold1/103074-7-4-3.wav", "Jack hammer")
]

# Make predictions before training
for file_path, true_label in test_files:
    predicted_label_before = make_predictions(model, le, file_path)
    old_predictions[file_path] = predicted_label_before

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model.fit(X_train, y_train, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs,
          validation_data=(X_test, y_test), verbose=1, callbacks=[wandb.keras.WandbCallback()])

# Make predictions after training
for file_path, true_label in test_files:
    predicted_label_after = make_predictions(model, le, file_path)
    wandb_table.add_data(file_path, true_label, old_predictions[file_path], predicted_label_after)

# Log the table to W&B
wandb.log({"Predictions": wandb_table})

wandb.finish()

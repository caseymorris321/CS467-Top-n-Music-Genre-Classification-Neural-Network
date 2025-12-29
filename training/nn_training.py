"""
This file contains the contents necessary to train the neural network

This code was inspired by the wandb_example.py file which can be found in the following link:
https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-Audio-Classification-with-Keras--Vmlldzo0MDQzNDUy
"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping


def load_data(data_path, metadata_path):
    """
    The load_data function will correctly load the training database such that the audio file is correctly related to
    it's associated data in an excel sheet.
    :param data_path: Path to the location of the audio files
    :param metadata_path: Excel sheet that contains the necessary data of each audio file
    :return: 2 arrays, one containing the feautures of the songs and one containing the labels
    """
    pass


if __name__ == "__main__":
    data_path = "DATA PATH HERE"
    metadata_path = "META DATA HERE"
    features, labels = load_data(data_path, metadata_path)

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels_onehot,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=labels_onehot
                                                        )

    # Create and compile the NN
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

    # Reshape the x data from array to Conv1D so that we can feed data into model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Setting batch size and epoch
    batch_size = 32
    epochs = 20

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

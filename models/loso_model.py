# CM

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

game_emotions = {1: 'calm', 2: 'boring', 3: 'funny', 4: 'horror'}

def make_windows(df, window_size=100, step_size=50):
    windows = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size].values
        windows.append(window)
    return np.array(windows)

# Organizing subjects for loso / normalizing data
subject_data = {}
for subject in range(1, 29):
    subject_str = f"{subject:02d}"
    all_subject_windows = []
    all_subject_labels = []
    for game in range(1, 5):
        path = rf"C:\eeg\Dataset - Emotion Recognition data Based on EEG Signals and Computer Games\Database for Emotion Recognition System Based on EEG Signals and Various Computer Games - GAMEEMO\GAMEEMO\(S{subject_str})\Preprocessed EEG Data\.csv format\S{subject_str}G{game}AllChannels.csv"
        try:
            game_data = pd.read_csv(path)
            game_data = game_data.drop(columns=['Unnamed: 14'], errors='ignore')
            game_data = game_data.fillna(method='ffill')
            game_data = (game_data - game_data.mean()) / game_data.std()
            windows = make_windows(game_data)
            labels = [game_emotions[game]] * len(windows)
            all_subject_windows.append(windows)
            all_subject_labels.extend(labels)
        except FileNotFoundError:
            continue
    if all_subject_windows:
        subject_data[subject] = (np.concatenate(all_subject_windows, axis=0), np.array(all_subject_labels))

# Label encoding
le = LabelEncoder()
all_labels = np.concatenate([labels for _, labels in subject_data.values()])
le.fit(all_labels)

# Masked autoencoder function
def masked_autoencoder(input_shape=(100, 14, 1), mask_ratio=0.25):
    inputs = layers.Input(shape=input_shape)

    # Generate mask (correct shape for data)
    mask = tf.cast(tf.random.uniform(shape=(1, 100, 14, 1)) > mask_ratio, dtype=tf.float32)
    masked_inputs = layers.Multiply()([inputs, mask])  # Ensure element-wise multiplication works

    # Encoder
    encoder = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    encoded = encoder(masked_inputs)

    # Decoder
    decoder = models.Sequential([
        layers.Dense(100 * 14, activation='relu'),
        layers.Reshape((100, 14, 1)),
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Ensure final shape matches input
    ])
    decoded = decoder(encoded)

    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')  # MSE for reconstruction

    return autoencoder, encoder

mae, encoder = masked_autoencoder()
mae.summary()

# Train autoencoder
mae, encoder = masked_autoencoder()
all_data = np.concatenate([data for data, _ in subject_data.values()], axis=0).reshape(-1, 100, 14, 1)
mae.fit(all_data, all_data, epochs=10, batch_size=32, verbose=1)

# Actual CNN model (with pretrained encoder from above)
def cnn_model(pretrained_encoder):
    inputs = layers.Input(shape=(100, 14, 1))
    features = pretrained_encoder(inputs)
    dense_layer = layers.Dense(64, activation='relu')(features)
    dropout_layer = layers.Dropout(0.3)(dense_layer)
    output_layer = layers.Dense(4, activation='softmax')(dropout_layer)
    model = models.Model(inputs, output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training/Testing
loso_accuracies = []
for test_subject in subject_data.keys():
    train_data = [data for subj, (data, _) in subject_data.items() if subj != test_subject]
    train_labels = [labels for subj, (_, labels) in subject_data.items() if subj != test_subject]
    if not train_data:
        continue
    X_train = np.concatenate(train_data, axis=0).reshape(-1, 100, 14, 1)
    y_train = np.concatenate(train_labels, axis=0)
    y_train = le.transform(y_train)
    y_train = to_categorical(y_train)
    
    X_test, y_test = subject_data[test_subject]
    X_test = X_test.reshape(-1, 100, 14, 1)
    y_test = le.transform(y_test)
    y_test = to_categorical(y_test)
    
    model = cnn_model(encoder)
    model.fit(X_train, y_train, epochs=10, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    loso_accuracies.append(test_accuracy)
    print(f"Subject {test_subject} Test Accuracy: {test_accuracy:.4f}")

# Print accuracies
if loso_accuracies:
    print(f"\nFinal LOSO Accuracy: {np.mean(loso_accuracies):.4f}")
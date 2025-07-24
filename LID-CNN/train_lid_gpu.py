# train_lid_gpu.py

import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving plots on the cluster
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, GRU, Dense, Dropout, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Parameters ---
# !!! IMPORTANT: Update this path to where you uploaded your data on Sharanga !!!
DATA_BASE_PATH = '/home/rahulkmr/ald_bhashini/audio'

LANGUAGE_FOLDER_MAP = {
    "Audio_hi": "hindi",
    "Audio_ml": "malayalam",
    "Audio_mr": "marathi",
    "Audio_pa": "punjabi"
}

SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 40
EPOCHS = 50
BATCH_SIZE = 64

# --- 1. Data Preparation ---
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
        target_length = DURATION * sample_rate
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

print("--> Starting Data Preparation...")
all_features = []
all_labels = []

for folder_name, language_label in LANGUAGE_FOLDER_MAP.items():
    lang_dir = os.path.join(DATA_BASE_PATH, folder_name)
    if os.path.isdir(lang_dir):
        print(f"Processing files for '{language_label}'...")
        file_list = sorted(os.listdir(lang_dir))
        for file_name in tqdm(file_list):
            if file_name.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(lang_dir, file_name)
                features = extract_features(file_path)
                if features is not None:
                    all_features.append(features)
                    all_labels.append(language_label)
    else:
        print(f"WARNING: Directory not found and skipped: {lang_dir}")

print(f"\nProcessed {len(all_features)} audio files.")
X = np.array(all_features)
y = np.array(all_labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
print("--> Data Preparation Complete.")

# --- 2. Model Building ---
print("\n--> Building CRNN Model...")
def build_crnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Reshape((-1, 256)), # Manually calculated based on input and pooling
        Bidirectional(GRU(256, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(GRU(128, return_sequences=False)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = X_train.shape[1:]
num_classes = len(label_encoder.classes_)
model = build_crnn_model(input_shape, num_classes)
model.summary()
print("--> Model Built Successfully.")

# --- 3. Model Training ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

print("\n--> Starting Model Training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)
print("\n--> Training Finished.")

# --- 4. Model Evaluation & Saving Artifacts ---
print("\n--> Evaluating Model and Saving Plots...")
# Save training history plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True)
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True)
plt.savefig('training_history.png')
print("Saved training history plot to training_history.png")

# Generate and save classification report and confusion matrix
y_pred = np.argmax(model.predict(X_val), axis=1)
report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
with open("classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
print("\nClassification Report saved to classification_report.txt")
print(report)

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix to confusion_matrix.png")

# --- 5. Save the Trained Model ---
model.save('language_identification_model.h5')
print("\n--> Model saved successfully as language_identification_model.h5")

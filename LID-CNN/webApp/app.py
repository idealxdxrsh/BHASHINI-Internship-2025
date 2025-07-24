import os
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, GRU, Dense, Dropout, Bidirectional
)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__)
CORS(app)  

# --- Model Definition (from your training script) ---
def build_crnn_model(input_shape, num_classes):
    """Defines the exact CRNN architecture used for training."""
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
        Reshape((-1, 256)), # Correctly reshapes CNN output for RNN
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
    return model


LANGUAGE_MAP = {
    0: "Hindi",
    1: "Malayalam",
    2: "Marathi",
    3: "Punjabi"
}
model = None
try:
    
    input_shape = (20, 216, 1) 
    num_classes = len(LANGUAGE_MAP)
    model = build_crnn_model(input_shape, num_classes)
    
    
    model.load_weights('language_identification_model.h5')
    print("* Model architecture defined and weights loaded successfully.")
except Exception as e:
    print(f"* Error loading model weights: {e}")


SAMPLE_RATE = 22050
DURATION = 5
N_MFCC = 20

def extract_features(audio_chunk, sample_rate):
    """Extracts MFCCs from a raw audio chunk."""
    try:
        mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=N_MFCC)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from chunk: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives an audio file, processes it, and returns the prediction.
    If the audio is longer than DURATION, it's split into chunks,
    and the prediction with the highest confidence is returned.
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
        
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file found in request.'}), 400

    audio_file = request.files['audio_file']
    
    # Save file temporarily to be loaded by librosa
    temp_filename = "temp_audio_file.wav"
    audio_file.save(temp_filename)

    # Load the entire audio file
    try:
        audio, sample_rate = librosa.load(temp_filename, sr=SAMPLE_RATE, res_type='kaiser_fast')
    except Exception as e:
        os.remove(temp_filename)
        return jsonify({'error': f'Could not load audio file: {e}'}), 500
    
    os.remove(temp_filename)

    chunk_length = DURATION * sample_rate
    
    # If audio is shorter than or equal to DURATION, process as before
    if len(audio) <= chunk_length:
        padded_audio = np.pad(audio, (0, chunk_length - len(audio)), 'constant')
        mfccs = extract_features(padded_audio, sample_rate)
        if mfccs is None:
            return jsonify({'error': 'Could not process the audio file.'}), 500
        
        mfccs_reshaped = mfccs[np.newaxis, ..., np.newaxis]
        prediction_probs = model.predict(mfccs_reshaped)[0]
        predicted_index = np.argmax(prediction_probs)
        predicted_language = LANGUAGE_MAP.get(predicted_index, "Unknown")
        confidence = np.max(prediction_probs) * 100
        
        return jsonify({
            'language': predicted_language,
            'confidence': confidence
        })

    # If audio is longer, slice it and find the best prediction
    else:
        best_prediction = {'language': 'Unknown', 'confidence': 0.0}
        
        # Iterate over the audio in chunks of DURATION seconds
        for i in range(0, len(audio) - chunk_length + 1, chunk_length):
            chunk = audio[i:i + chunk_length]
            
            mfccs = extract_features(chunk, sample_rate)
            if mfccs is None:
                continue # Skip chunk if feature extraction fails

            mfccs_reshaped = mfccs[np.newaxis, ..., np.newaxis]
            prediction_probs = model.predict(mfccs_reshaped)[0]
            
            current_confidence = np.max(prediction_probs) * 100
            
            if current_confidence > best_prediction['confidence']:
                predicted_index = np.argmax(prediction_probs)
                best_prediction['language'] = LANGUAGE_MAP.get(predicted_index, "Unknown")
                best_prediction['confidence'] = float(current_confidence)

        if best_prediction['confidence'] == 0.0:
             return jsonify({'error': 'Could not get a confident prediction from any audio chunk.'}), 500

        return jsonify(best_prediction)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

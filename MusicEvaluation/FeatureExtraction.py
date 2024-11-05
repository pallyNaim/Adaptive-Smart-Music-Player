import librosa
import numpy as np
import pandas as pd
import os

def extract_features(file_name):
    signal, sr = librosa.load(file_name)
    
    features = {
        'file_name': file_name,
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(signal)),
        'rmse': np.mean(librosa.feature.rms(y=signal)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)),
        'spectral_contrast': np.mean(librosa.feature.spectral_contrast(y=signal, sr=sr)),
        'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)),
        'tempo': librosa.beat.tempo(y=signal, sr=sr)[0],
        'harmonic': np.mean(librosa.effects.harmonic(signal)),
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=signal, sr=sr)),
        'tonnetz': np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)),
    }
    
    # Extracting MFCCs and adding them as individual features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    for i in range(1, 14):
        features[f'mfcc_{i}'] = np.mean(mfccs[i-1])
    
    return features

def append_to_csv(features, csv_file='music_features_new.csv'):
    # Convert the features dictionary to a DataFrame
    features_df = pd.DataFrame([features])
    
    # Check if the CSV file exists
    if not os.path.isfile(csv_file):
        features_df.to_csv(csv_file, index=False)
    else:
        features_df.to_csv(csv_file, mode='a', header=False, index=False)

def process_directory(directory, csv_file='music_features_new.csv'):
    # Loop through each file in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.mp3'):  # Process only mp3 files
            full_path = os.path.join(directory, file_name)
            features = extract_features(full_path)
            append_to_csv(features, csv_file=csv_file)

# Specify the directory containing the sound sample files
directory = 'Song/newSong'

# Process all files in the directory
process_directory(directory, csv_file='music_features_newSong.csv')

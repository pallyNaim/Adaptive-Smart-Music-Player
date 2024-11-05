import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
import pygame  # Importing pygame for music playback
from emotion_video_classifier import emotion_testing  # Importing the emotion testing function

# Initialize Pygame mixer
pygame.mixer.init()

# Load the dataset
data = pd.read_csv(r'C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\Song Dataset\data.csv\datav6.csv')

# Display the first few rows of the dataset to verify
print(data.head())

# Specify the columns to use for clustering
col_features = ['zero_crossing_rate', 'mfcc_11', 'chroma_stft', 'spectral_contrast']

# Normalize the features
X = MinMaxScaler().fit_transform(data[col_features])

# Perform KMeans clustering
kmeans = KMeans(init="k-means++", n_clusters=2, random_state=15).fit(X)
data['kmeans'] = kmeans.labels_

# Get recommendations based on emotion
def get_results(emotion_code):
    NUM_RECOMMEND = 10
    if emotion_code == 0:
        return data[data['kmeans'] == 0][['file_name', 'full_path']].head(NUM_RECOMMEND)  # Return file_name and full_path
    else:
        return data[data['kmeans'] == 1][['file_name', 'full_path']].head(NUM_RECOMMEND)  # Return file_name and full_path

# Function to play music using the full path
def play_music(file_path):
    try:
        pygame.mixer.music.load(file_path)  # Load the song using its full path
        pygame.mixer.music.play()
    except pygame.error as e:
        print(f"Error playing file {file_path}: {e}")

# Function to stop music
def stop_music():
    pygame.mixer.music.stop()

# Function to display results
def final(emotion_code):
    root1 = tk.Tk()
    root1.title("Your Playlist")
    root1.geometry("600x400")
    root1.configure(bg='black')

    # Frame for the playlist
    frame = tk.Frame(root1, bg='black')
    frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    recommended_songs = get_results(emotion_code)

    # Create Treeview with Scrollbar
    tree_scroll = tk.Scrollbar(frame)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    tree = ttk.Treeview(frame, yscrollcommand=tree_scroll.set, selectmode="extended", height=10)
    tree.pack()

    tree_scroll.config(command=tree.yview)

    # Define Columns
    tree['columns'] = ('Track Name', 'Full Path')

    # Format Columns
    tree.column("#0", width=0, stretch=tk.NO)
    tree.column("Track Name", anchor=tk.W, width=400)
    tree.column("Full Path", anchor=tk.W, width=0)  # Hide full path but store it for playing

    # Create Headings
    tree.heading("#0", text="", anchor=tk.W)
    tree.heading("Track Name", text="Track Name", anchor=tk.W)
    tree.heading("Full Path", text="Full Path", anchor=tk.W)

    # Insert Data into Treeview
    for _, row in recommended_songs.iterrows():
        song_name = row['file_name']  # Display the file name (not the full path)
        full_path = row['full_path']  # Store full path to use for playing
        tree.insert(parent='', index='end', text="", values=(song_name, full_path))

    # Play Button
    def play_selected_song():
        selected_item = tree.selection()
        if selected_item:
            full_path = tree.item(selected_item, 'values')[1]  # Get the full path from the hidden column
            play_music(full_path)

    play_button = tk.Button(root1, text="Play", command=play_selected_song)
    play_button.pack(pady=5)

    # Stop Button
    stop_button = tk.Button(root1, text="Stop", command=stop_music)
    stop_button.pack(pady=5)

    root1.mainloop()

# Emotion detection and display
emotion_word = emotion_testing()  # Get the detected emotion from the imported function

# Map emotion to code (adjust according to your emotion categories)
if emotion_word == 'sad':
    emotion_code = 0
elif emotion_word == 'happy':
    emotion_code = 1
else:
    emotion_code = 1  # Default to a specific cluster or handle other emotions accordingly

# Display final playlist based on emotion
final(emotion_code)

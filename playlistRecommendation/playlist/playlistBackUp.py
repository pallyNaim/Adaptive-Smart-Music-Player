#Before add link t


from tkinter import ttk
from PIL import Image, ImageTk
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from emotion_video_classifier import emotion_testing
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import messagebox

# Spotify credentials
client_id = 'd7c40eb156434decac8ea55adfcff15d'
client_secret = '76c9c8865c4e4bc3a894b97577b0b6fe'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

root = tk.Tk()
root.title('Song Recommendation')
root.geometry("600x400")
root.configure(bg='black')

song_name_var = tk.StringVar()

photo = ImageTk.PhotoImage(file=r"C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\image\musicback.jpg")
l = tk.Label(root, image=photo)
l.image = photo
l.grid()

def submit():
    global song_name
    song_name = song_entry.get()
    messagebox.showinfo("Information", "Fetching songs with similar features...")
    root.destroy()

# Input fields for song name
song_label = tk.Label(root, text='Enter Song Name', font=('calibre', 10, 'bold'), bg='black', fg='white')
song_entry = tk.Entry(root, textvariable=song_name_var, font=('calibre', 10, 'normal'))
sub_btn = tk.Button(root, text='Submit', command=submit)

song_label.grid(row=0, column=0)
song_entry.grid(row=1, column=0)
sub_btn.grid(row=2, column=0)
root.mainloop()

# Search for the song on Spotify
result = sp.search(song_name, type='track')
track = result['tracks']['items'][0]
track_id = track['id']

# Get audio features of the input song
input_features = sp.audio_features(track_id)[0]
input_features_df = pd.DataFrame([input_features])

# Prepare your existing dataset with songs and their features
df = pd.read_csv(r'C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\Song Dataset\data.csv\data.csv')
df.drop_duplicates(inplace=True, subset=['name'])

# Define the features for similarity comparison and clustering
feature_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
cluster_columns = ['danceability', 'energy', 'valence', 'loudness']

# Normalize both the input song features and the dataset features for comparison
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_columns]), columns=feature_columns)
input_features_scaled = scaler.transform(input_features_df[feature_columns])

# Calculate cosine similarity between the input song and all songs in the dataset
similarities = cosine_similarity(df_scaled, input_features_scaled)

# Add similarity scores to the dataset
df['similarity'] = similarities

# Perform clustering for emotion-based recommendation
X = MinMaxScaler().fit_transform(df[cluster_columns])
kmeans = KMeans(n_clusters=2, random_state=15).fit(X)
df['kmeans'] = kmeans.labels_

# Define a function to fetch recommendations based on emotion code
def get_results(emotion_code):
    NUM_RECOMMEND = 10
    if emotion_code == 0:
        return df[df['kmeans'] == 0].sort_values(by='similarity', ascending=False).head(NUM_RECOMMEND)
    else:
        return df[df['kmeans'] == 1].sort_values(by='similarity', ascending=False).head(NUM_RECOMMEND)



emotion_word = emotion_testing()
if emotion_word == 'sad':
    emotion_code = 0
else:
    emotion_code = 1

# Get final recommendations based on the emotion
recommended_songs = get_results(emotion_code)

# Function to display recommended songs in a new window
def final_recommendations(recommended_songs):
    root1 = tk.Tk()
    root1.title("Your Song Recommendations")
    root1.geometry("600x400")
    root1.configure(bg='black')

    frame = tk.Frame(root1, bg='black')
    frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    # Create Treeview with Scrollbar
    tree_scroll = tk.Scrollbar(frame)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    tree = ttk.Treeview(frame, yscrollcommand=tree_scroll.set, selectmode="extended", height=10)
    tree.pack()

    tree_scroll.config(command=tree.yview)

    # Define Columns
    tree['columns'] = ('Track Name',)

    # Format Columns
    tree.column("#0", width=0, stretch=tk.NO)  # Remove the default column
    tree.column("Track Name", anchor=tk.W, width=400)

    # Create Headings
    tree.heading("#0", text="", anchor=tk.W)
    tree.heading("Track Name", text="Track Name", anchor=tk.W)

    # Insert Data into Treeview
    for index, song_name in recommended_songs['name'].iteritems():
        tree.insert(parent='', index='end', iid=index, text="", values=(song_name,))

    root1.mainloop()

# Display final recommendations
final_recommendations(recommended_songs)

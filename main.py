import dearpygui.dearpygui as dpg
import ntpath
import json
from mutagen.mp3 import MP3
from tkinter import Tk, filedialog
import threading
import pygame
import time
import random
import os
import atexit

dpg.create_context()
dpg.create_viewport(title="CLPMP Music", large_icon="icon.ico", small_icon="icon.ico")
pygame.mixer.init()
global state
state = None

global no
no = 0

_DEFAULT_MUSIC_VOLUME = 0.5
pygame.mixer.music.set_volume(0.5)

music_directory = r"C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\Music"

def update_volume(sender, app_data):
    pygame.mixer.music.set_volume(app_data / 100.0)

def load_database():
    for filename in os.listdir(music_directory):
        if filename.endswith(".mp3"):
            full_path = os.path.join(music_directory, filename)
            dpg.add_button(label=f"{ntpath.basename(filename)}", callback=play, width=-1,
                           height=25, user_data=full_path.replace("\\", "/"), parent="list")
            dpg.add_spacer(height=2, parent="list")

#def update_database(filename: str):
#    data = json.load(open("data/songs.json", "r+"))
#    if filename not in data["songs"]:
#        data["songs"] += [filename]
#    json.dump(data, open("data/songs.json", "w"), indent=4)

def update_slider():
    global state
    while pygame.mixer.music.get_busy() or state != 'paused':
        dpg.configure_item(item="pos", default_value=pygame.mixer.music.get_pos() / 1000)
        time.sleep(0.7)
    state = None
    dpg.configure_item("cstate", default_value=f"State: None")
    dpg.configure_item("csong", default_value="Now Playing : ")
    dpg.configure_item("play", label="Play")
    dpg.configure_item(item="pos", max_value=100)
    dpg.configure_item(item="pos", default_value=0)

def play(sender, app_data, user_data):
    global state, no
    if user_data:
        no = user_data  # Set the current playing song path
        pygame.mixer.music.load(user_data)
        audio = MP3(user_data)
        dpg.configure_item(item="pos", max_value=audio.info.length)
        pygame.mixer.music.play()
        thread = threading.Thread(target=update_slider, daemon=False).start()
        if pygame.mixer.music.get_busy():
            dpg.configure_item("play", label="Pause")
            state = "playing"
            dpg.configure_item("cstate", default_value=f"State: Playing")
            dpg.configure_item("csong", default_value=f"Now Playing : {ntpath.basename(user_data)}")


def play_pause():
    global state, no
    if state == "playing":
        state = "paused"
        pygame.mixer.music.pause()
        dpg.configure_item("play", label="Play")
        dpg.configure_item("cstate", default_value=f"State: Paused")
    elif state == "paused":
        state = "playing"
        pygame.mixer.music.unpause()
        dpg.configure_item("play", label="Pause")
        dpg.configure_item("cstate", default_value=f"State: Playing")
    else:
        songs = [os.path.join(music_directory, f) for f in os.listdir(music_directory) if f.endswith(".mp3")]
        if songs:
            song = random.choice(songs)
            no = song
            pygame.mixer.music.load(song)
            pygame.mixer.music.play()
            thread = threading.Thread(target=update_slider, daemon=False).start()
            dpg.configure_item("play", label="Pause")
            if pygame.mixer.music.get_busy():
                audio = MP3(song)
                dpg.configure_item(item="pos", max_value=audio.info.length)
                state = "playing"
                dpg.configure_item("csong", default_value=f"Now Playing : {ntpath.basename(song)}")
                dpg.configure_item("cstate", default_value=f"State: Playing")

# Add this import at the beginning of your script
import subprocess
# Function to launch facial.py
def launch_facial_detection():
    subprocess.Popen(["python", "facial.py"])
def recommendation():
    subprocess.Popen(["python", "playlistMain.py"])         

##################################################################################################################
def pre(sender, app_data, user_data):
    global state, no
    songs = [os.path.join(music_directory, f) for f in os.listdir(music_directory) if f.endswith(".mp3")]
    try:
        n = songs.index(no)
        if n == 0:
            n = len(songs)
        no = songs[n - 1]
        play(sender, app_data, no)
    except ValueError:
        pass

def next(sender, app_data, user_data):
    global state, no
    songs = [os.path.join(music_directory, f) for f in os.listdir(music_directory) if f.endswith(".mp3")]
    try:
        n = songs.index(no)
        if n == len(songs) - 1:
            n = -1
        no = songs[n + 1]
        play(sender, app_data, no)
    except ValueError:
        pass


def stop():
    global state
    pygame.mixer.music.stop()
    state = None

def add_files():
    root = Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(filetypes=[("Music Files", ("*.mp3", "*.wav", "*.ogg"))])
    root.quit()
    if filename.endswith(".mp3" or ".wav" or ".ogg"):
        dpg.add_button(label=f"{ntpath.basename(filename)}", callback=play, width=-1, height=25,
                       user_data=filename.replace("\\", "/"), parent="list")
        dpg.add_spacer(height=2, parent="list")

def add_folder():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    root.quit()
    for filename in os.listdir(folder):
        if filename.endswith(".mp3" or ".wav" or ".ogg"):
            dpg.add_button(label=f"{ntpath.basename(filename)}", callback=play, width=-1, height=25,
                           user_data=os.path.join(folder, filename).replace("\\", "/"), parent="list")
            dpg.add_spacer(height=2, parent="list")

def search(sender, app_data, user_data):
    dpg.delete_item("list", children_only=True)
    for filename in os.listdir(music_directory):
        if filename.endswith(".mp3") and app_data.lower() in filename.lower():
            full_path = os.path.join(music_directory, filename)
            dpg.add_button(label=f"{ntpath.basename(filename)}", callback=play, width=-1,
                           height=25, user_data=full_path.replace("\\", "/"), parent="list")
            dpg.add_spacer(height=2, parent="list")

def removeall():
    dpg.delete_item("list", children_only=True)
    load_database()
##################################################################################################################

with dpg.theme(tag="base"):
    with dpg.theme_component():
        dpg.add_theme_color(dpg.mvThemeCol_Button, (130, 142, 250))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (137, 142, 255, 95))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (137, 142, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 4)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4, 4)
        dpg.add_theme_style(dpg.mvStyleVar_WindowTitleAlign, 0.50, 0.50)
        dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 14)
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (25, 25, 25))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (0, 0, 0, 0))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (130, 142, 250))
        dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (221, 166, 185))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (172, 174, 197))

with dpg.theme(tag="slider_thin"):
    with dpg.theme_component():
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (130, 142, 250, 99))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (172, 174, 197))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (172, 174, 197))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (172, 174, 197))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (130, 142, 250))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (130, 142, 250, 66))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (130, 142, 250, 66))

with dpg.window(tag="Main", width=345, height=450, no_move=True, no_resize=True):
    with dpg.child_window(autosize_x=True, autosize_y=True, menubar=True, border=False):
        with dpg.menu_bar():
            with dpg.menu(label="Settings"):
                ###########################################################################################
                dpg.add_menu_item(label="Open Concentration Monitoring", callback=launch_facial_detection)
                dpg.add_menu_item(label="Create playlist", callback=recommendation)
                ###########################################################################################
                with dpg.menu(label="Add Files"):
                    dpg.add_menu_item(label="Add Files", callback=add_files)
                    dpg.add_menu_item(label="Add Folder", callback=add_folder)
                dpg.add_menu_item(label="Remove all", callback=removeall)
        with dpg.child_window(autosize_x=True, height=100, border=False):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Previous", callback=pre)
                dpg.add_button(label="Play", callback=play_pause, tag="play")
                dpg.add_button(label="Stop", callback=stop)
                dpg.add_button(label="Next", callback=next)
            dpg.add_slider_float(label="", tag="pos", width=-1, default_value=0)
            dpg.add_slider_float(label="", width=-1, min_value=0, max_value=100, default_value=50,
                                 callback=update_volume)
        with dpg.group(horizontal=True):
            dpg.add_text("State: None", tag="cstate", color=(200, 205, 255, 255))
            dpg.add_spacer(width=80)
            dpg.add_text("Now Playing : ", tag="csong", color=(200, 205, 255, 255))
        dpg.add_spacer(height=2)
        dpg.add_input_text(label="Search", width=-1, callback=search, user_data=[], hint="Search...")
        with dpg.child_window(autosize_x=True, border=False, tag="list"):
            load_database()

dpg.bind_theme("base")
dpg.bind_item_theme("pos", "slider_thin")
dpg.set_primary_window("Main", True)
dpg.create_viewport(width=350, height=450)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

atexit.register(stop)

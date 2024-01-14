import pandas as pd
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify, render_template, request
import tensorflow_hub as hub
import os
from pygame import mixer
import requests

nltk.download('vader_lexicon')


app = Flask(__name__, template_folder='')

# Read the dataset from a CSV file
dataset_path = 'Songs.csv'
df = pd.read_csv(dataset_path)

def add_emotion_column(df, sia):
    # Add a new column 'Emotion' to store the predicted emotion
    df['Emotion'] = df['Lyrics'].apply(lambda x: get_emotion_from_lyrics(x, sia))

# Assuming you have a function to get recommendations based on emotions or artist
def get_recommendations(song_title, df, sia):
    # Find the song in the DataFrame
    result = df[df['Title'].str.lower() == song_title.lower()]

    if not result.empty:
        # Get the emotion for the searched song
        emotion_of_song = get_emotion_from_lyrics(result['Lyrics'].values[0], sia)

        # Get all songs with the same emotion (excluding the searched song)
        recommendations = df[df['Title'].str.lower() != song_title.lower()]
        recommendations = recommendations[recommendations.apply(lambda row: get_emotion_from_lyrics(row['Lyrics'], sia), axis=1) == emotion_of_song]

        # Prepare the recommendations as a list of dictionaries
        recommendations_list = []
        for index, row in recommendations.iterrows():
            recommendations_list.append({'title': row['Title'], 'artist': row['Artist'], 'emotion': get_emotion_from_lyrics(row['Lyrics'], sia)})

        return recommendations_list
    else:
        return None


def get_emotion_from_lyrics(lyrics, sia):

    polarity_score = sia.polarity_scores(lyrics)['compound']

    # Classify genres based on sentiment polarity scores
    if polarity_score >= 0.6:
        return "Dance"
    elif polarity_score <= -0.5:
        return "Dark"
    elif -0.1 < polarity_score < 0.1:
        return "Neutral"
    elif 0.25 > polarity_score > 0.2:
        return "Heart Broken"
    elif -0.25 < polarity_score < -0.2:
        return "Sad"
    else:
        return "Other"
    
def search_song(song_name, df, sia):
    sia = SentimentIntensityAnalyzer()
    result = df[df['Title'].str.lower() == song_name.lower()]

    if not result.empty:
        polarity_score = sia.polarity_scores(result['Lyrics'].values[0])['compound']
        emotion = get_emotion_from_lyrics(result['Lyrics'].values[0], sia)
        artist = result['Artist'].values[0]

        return {
            'Title': result['Title'].values[0],
            'Artist': artist,
            'Emotion': emotion
        }
    else:
        return None
    
def search_emotion(emotion, df, sia):

    result = df[df['Emotion'].str.lower() == emotion.lower()]

    if not result.empty:
        songs_info = []
        for index, row in result.iterrows():
            polarity_score = sia.polarity_scores(row['Lyrics'])['compound']
            emotions = get_emotion_from_lyrics(row['Lyrics'], sia)

            song_info = {
                'Title': row['Title'],
                'Artist': row['Artist'],
                'Emotion': emotions
            }
            songs_info.append(song_info)

        return songs_info
    else:
        return None
    
#-----------------------------------------------------Music--------------------------------
# Instantiate mixer
mixer.init()


def check_music_exists(track_name):
    track_exists_check = track_name + '.mp3'
    print(track_exists_check, "track name")
    if os.path.exists(os.getcwd()+'/'+track_exists_check):
        mixer.music.load(track_exists_check)
        return True

    return False


def search_for_music(track_name):

    # Replace 'YOUR_CLIENT_ID' and 'YOUR_CLIENT_SECRET' with your Spotify API credentials
    client_id = 'ba032a5182044df09ddb3a5128f0c848'
    client_secret = 'df162ca97a6049a4bfe20bab8b877463'

    # Replace 'YOUR_TRACK_NAME' with the name of the track you want to play

    # Step 1: Authenticate with Spotify API and get an access token
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_data = {'grant_type': 'client_credentials'}
    auth_response = requests.post(auth_url, auth=(
        client_id, client_secret), data=auth_data)
    auth_response_data = auth_response.json()
    access_token = auth_response_data['access_token']

    # Step 2: Search for the track
    search_url = f'https://api.spotify.com/v1/search?q={track_name}&type=track&limit=1'
    search_headers = {'Authorization': f'Bearer {access_token}'}
    search_response = requests.get(search_url, headers=search_headers)

    search_response_data = search_response.json()
    # Step 3: Get the preview URL
    if 'tracks' in search_response_data and 'items' in search_response_data['tracks'] and search_response_data['tracks']['items']:
        track = search_response_data['tracks']['items'][0]
        preview_url = track.get('preview_url')

        if preview_url:
            print("has preview")
        else:
            print('No preview available for this track.')
    else:
        print('Track not found.')

    def download_mp3(url, output_file):
        response = requests.get(url)
        print(response, " outfile")

        with open(output_file, 'wb') as f:
            f.write(response.content)

    # Replace 'your_url_here' with the actual URL of the MP3 file
    mp3_url = preview_url
    output_filename = track_name+".mp3"

    download_mp3(mp3_url, output_filename)
    mixer.music.load(output_filename)

@app.route('/')
def index():
# Default values

    encoded_dance_image = None
    encoded_happy_image = None
    encoded_image = None

    # Get user input from the query parameters
    user_input = request.args.get('emotion', '')
    print(user_input,"i npuuttttttttttttttttttttttttttttttt")

    # if user_input == 'dance':
    #     print("herererrerr")
    #     content_dance_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Two_dancers.jpg/640px-Two_dancers.jpg' 
    #     style_dance_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Contemporary_abstract_painting_by_Ib_Benoh_1970s.jpg/640px-Contemporary_abstract_painting_by_Ib_Benoh_1970s.jpg' 

    #     content_happy_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Monkey-accordion.jpg/640px-Monkey-accordion.jpg'
    #     style_happy_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Mandelbrot_Set_Image_09_by_Aokoroko.jpg/640px-Mandelbrot_Set_Image_09_by_Aokoroko.jpg'

    #     output_image_size = 384  

    #     content_img_size = (output_image_size, output_image_size)
    #     style_img_size = (256, 256) 

    #     content_dance = load_image(content_dance_url, content_img_size)
    #     style_dance = load_image(style_dance_url, style_img_size)
    #     style_dance = tf.nn.avg_pool(style_dance, ksize=[3,3], strides=[1,1], padding='SAME')

    #     content_happy = load_image(content_happy_url, content_img_size)
    #     style_happy = load_image(style_happy_url, style_img_size)
    #     style_happy = tf.nn.avg_pool(style_happy, ksize=[3,3], strides=[1,1], padding='SAME')
    
    #     # Load TF Hub module.
    #     hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    #     hub_module = hub.load(hub_handle)

    #     danceOutputs = hub_module(content_dance, style_dance)
    #     stylized_danceImage = danceOutputs[0]
    #     danceOutputs = hub_module(tf.constant(content_dance), tf.constant(style_dance))
    #     stylized_danceImage = danceOutputs[0]

    #     happyOutputs = hub_module(content_happy, style_happy)
    #     stylized_happyImage = danceOutputs[0]
    #     happyOutputs = hub_module(tf.constant(content_happy), tf.constant(style_happy))
    #     stylized_happyImage = happyOutputs[0]

    #     # Get the base64-encoded image string
    #     encoded_image = show_and_return_base64([stylized_danceImage, stylized_happyImage],
    #                                    titles=['Dance', 'Happy'])
    #     print(encoded_image)
    #     return render_template('index.html', encoded_image=encoded_image)
    # elif encoded_image==None:
    #     return render_template('index.html', encoded_image=encoded_image)

    
    #     print("hererererer")
    #     content_dance_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Two_dancers.jpg/640px-Two_dancers.jpg' 
    #     style_dance_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Contemporary_abstract_painting_by_Ib_Benoh_1970s.jpg/640px-Contemporary_abstract_painting_by_Ib_Benoh_1970s.jpg' 

    #     output_image_size = 384  
    #     content_img_size = (output_image_size, output_image_size)
    #     style_img_size = (256, 256) 

    #     content_dance = load_image(content_dance_url, content_img_size)
    #     style_dance = load_image(style_dance_url, style_img_size)
    #     style_dance = tf.nn.avg_pool(style_dance, ksize=[3,3], strides=[1,1], padding='SAME')

    #     # Load TF Hub module.
    #     hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    #     hub_module = hub.load(hub_handle)

    #     # Dance Stylization
    #     danceOutputs = hub_module(content_dance, style_dance)
    #     stylized_danceImage = danceOutputs[0]
    #     print("it is coming here")
    #     danceOutputs = hub_module(tf.constant(content_dance), tf.constant(style_dance))
    #     stylized_danceImage = danceOutputs[0]
    #     print(stylized_danceImage, " immmmmmmggggggg")
    #     # Get the base64-encoded image string
    #     #encoded_dance_image = show_and_return_base64([stylized_danceImage], titles=['Dance'])
    #     encoded_image = show_and_return_base64([stylized_danceImage], titles=['Dance'])
    #     print(encoded_image)
    # elif user_input == 'happy':
    #     content_happy_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Monkey-accordion.jpg/640px-Monkey-accordion.jpg'
    #     style_happy_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Mandelbrot_Set_Image_09_by_Aokoroko.jpg/640px-Mandelbrot_Set_Image_09_by_Aokoroko.jpg'

    #     output_image_size = 384  
    #     content_img_size = (output_image_size, output_image_size)
    #     style_img_size = (256, 256) 

    #     content_happy = load_image(content_happy_url, content_img_size)
    #     style_happy = load_image(style_happy_url, style_img_size)
    #     style_happy = tf.nn.avg_pool(style_happy, ksize=[3,3], strides=[1,1], padding='SAME')

    #     # Load TF Hub module.
    #     hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    #     hub_module = hub.load(hub_handle)

    #     # Happy Stylization
    #     happy_outputs = hub_module(content_happy, style_happy)
    #     stylized_happy_image = happy_outputs[0]

    #     # Get the base64-encoded image string
    #     #encoded_happy_image = show_and_return_base64([stylized_happy_image], titles=['Happy'])
    #     encoded_image = show_and_return_base64([stylized_happy_image], titles=['Happy'])
        
        #working
    content_dance_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Two_dancers.jpg/640px-Two_dancers.jpg' 
    style_dance_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Contemporary_abstract_painting_by_Ib_Benoh_1970s.jpg/640px-Contemporary_abstract_painting_by_Ib_Benoh_1970s.jpg' 

    content_happy_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Happy_smiley_face.png/640px-Happy_smiley_face.png'
    style_happy_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Illuminated_Ferris_wheel%2C_bouncing_castle_and_carousel_at_night_in_a_funfair_in_Vientiane%2C_Laos.jpg/640px-Illuminated_Ferris_wheel%2C_bouncing_castle_and_carousel_at_night_in_a_funfair_in_Vientiane%2C_Laos.jpg'

    content_heartBroken_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/The_Far_Cry_%281926%29_-_1.jpg/640px-The_Far_Cry_%281926%29_-_1.jpg'
    style_heartBroken_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Love_Heart_broken_enameled.svg/640px-Love_Heart_broken_enameled.svg.png' 

    content_dark_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0b/Isolation_and_Community.jpg'
    style_dark_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Haunts_of_solitude_%288329925989%29.jpg/640px-Haunts_of_solitude_%288329925989%29.jpg'

    content_sad_url = 'https://upload.wikimedia.org/wikipedia/commons/d/d0/Thoma_Loneliness.jpg'
    style_sad_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Alphonse_Osbert_-_La_Solitude_du_Christ.jpg/640px-Alphonse_Osbert_-_La_Solitude_du_Christ.jpg'

    output_image_size = 384  

    content_img_size = (output_image_size, output_image_size)
    style_img_size = (256, 256) 

    content_dance = load_image(content_dance_url, content_img_size)
    style_dance = load_image(style_dance_url, style_img_size)
    style_dance = tf.nn.avg_pool(style_dance, ksize=[3,3], strides=[1,1], padding='SAME')

    content_happy = load_image(content_happy_url, content_img_size)
    style_happy = load_image(style_happy_url, style_img_size)
    style_happy = tf.nn.avg_pool(style_happy, ksize=[3,3], strides=[1,1], padding='SAME')

    content_heartBroken = load_image(content_heartBroken_url, content_img_size)
    style_heartBroken = load_image(style_heartBroken_url, style_img_size)
    style_heartBroken = tf.nn.avg_pool(style_heartBroken, ksize=[3,3], strides=[1,1], padding='SAME')

    content_dark = load_image(content_dark_url, content_img_size)
    style_dark = load_image(style_dark_url, style_img_size)
    style_dark = tf.nn.avg_pool(style_dark, ksize=[3,3], strides=[1,1], padding='SAME')

    content_sad = load_image(content_sad_url, content_img_size)
    style_sad = load_image(style_sad_url, style_img_size)
    style_sad = tf.nn.avg_pool(style_sad, ksize=[3,3], strides=[1,1], padding='SAME')
    
    # Load TF Hub module.
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    danceOutputs = hub_module(content_dance, style_dance)
    stylized_danceImage = danceOutputs[0]
    danceOutputs = hub_module(tf.constant(content_dance), tf.constant(style_dance))
    stylized_danceImage = danceOutputs[0]

    happyOutputs = hub_module(content_happy, style_happy)
    stylized_happyImage = danceOutputs[0]
    happyOutputs = hub_module(tf.constant(content_happy), tf.constant(style_happy))
    stylized_happyImage = happyOutputs[0]

    heartBrokenOutputs = hub_module(content_heartBroken, style_heartBroken)
    stylized_heartBrokenImage = heartBrokenOutputs[0]
    heartBrokenOutputs = hub_module(tf.constant(content_heartBroken), tf.constant(style_heartBroken))
    stylized_heartBrokenImage = heartBrokenOutputs[0]

    darkOutputs = hub_module(content_dark, style_dark)
    stylized_darkImage = darkOutputs[0]
    darkOutputs = hub_module(tf.constant(content_dark), tf.constant(style_dark))
    stylized_darkImage = darkOutputs[0]

    sadOutputs = hub_module(content_sad, style_sad)
    stylized_sadImage = sadOutputs[0]
    sadOutputs = hub_module(tf.constant(content_sad), tf.constant(style_sad))
    stylized_sadImage = sadOutputs[0]

    # Get the base64-encoded image string
    encoded_image = show_and_return_base64([stylized_danceImage, stylized_happyImage, stylized_heartBrokenImage, stylized_darkImage, stylized_sadImage],
                                       titles=['Dance', 'Happy', 'Heart Broken', 'Dark', 'Sad'])
    
    return render_template('index.html', encoded_image=encoded_image)

@app.route('/play_song')
def give_song():
    user_input_song = request.args.get('song_name', '')
    print(user_input_song, " input")
    check_truth = check_music_exists(user_input_song)
    if check_truth:
        mixer.music.play(1)
    else:
        search_for_music(user_input_song)
        mixer.music.play(1)

    return jsonify({'status': 'play'})


@app.route('/pause')
def pause_music():
    mixer.music.pause()
    return jsonify({'status': 'paused'})


@app.route('/resume')
def resume_music():
    print("here")
    mixer.music.unpause()
    return jsonify({'status': 'resumed'})


@app.route('/stop')
def stop_music():
    mixer.music.stop()
    return jsonify({'status': 'stopped'})
@app.route('/get_emotions')
def get_emotions():
    # Load the dataset from the CSV file
    dataset_path = 'Songs.csv'
    df = pd.read_csv(dataset_path)

    sia = SentimentIntensityAnalyzer()

    # Add 'Emotion' column
    add_emotion_column(df, sia)

    # Prepare the result as JSON
    result = df[['Title', 'Emotion']].to_dict(orient='records')

    return jsonify(result)

@app.route('/search_song')
def search_song_route():
    # Get the user-inputted song name from the query parameters
    user_input_song = request.args.get('song_name', '')
    sia = SentimentIntensityAnalyzer()

    # Perform the search and get the result
    search_result = search_song(user_input_song, df, sia)

    if search_result:
         # Get recommendations based on the searched song
        recommendations = get_recommendations(search_result['Title'], df, sia)
        if recommendations:
            return jsonify({'search_results': search_result, 'recommendations': recommendations})
        else:
            return jsonify({'search_results': search_result, 'recommendations': [], 'error': 'No recommendations found'})    
    else:
        return jsonify({'error': 'Song not found'})

@app.route('/search_by_emotion')
def search_by_emotion_route():

    user_input_emotion = request.args.get('emotion', '')

    sia = SentimentIntensityAnalyzer()

    # Check if 'Emotion' column exists in the DataFrame
    if 'Emotion' not in df.columns:
        # If not, create the 'Emotion' column using your function
        add_emotion_column(df, sia)

    # Filter songs by emotion
    # result = df[df['Emotion'].str.lower() == user_input_emotion.lower()]
    
    search_result = search_emotion(user_input_emotion, df, sia)

    print("search results", search_result)

    if search_result:
        return jsonify({'search_results': search_result})
    else:
        return jsonify({'error': f'No songs found for the specified emotion: {user_input_emotion}'})
    
import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub
from io import BytesIO
import base64

print("TF Version: ", tf.__version__)
# print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# @title Define image loading and visualization functions  { display-mode: "form" }

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_and_return_base64(images, titles=('', '')):
    n = len(images)
    image_sizes = [image.shape[1] for image in images]
    w = (image_sizes[0] * 6) // 320
    fig = plt.figure(figsize=(w * n, w))
    gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
    for i in range(n):
        ax = fig.add_subplot(gs[i])
        ax.imshow(images[i][0], aspect='equal')
        ax.axis('off')
        ax.set_title(titles[i] if len(titles) > i else '')

    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image as base64
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
  
    plt.close(fig)  # Close the figure to free up resources

    return encoded_image

if __name__ == '__main__':
    app.run(debug=True,port=3000)



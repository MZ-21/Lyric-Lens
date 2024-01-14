import pandas as pd
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify, render_template, request
import tensorflow_hub as hub

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

@app.route('/')
def index():
    content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
    style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'  # @param {type:"string"}
    output_image_size = 384  # @param {type:"integer"}

    # The content image size can be arbitrary.
    content_img_size = (output_image_size, output_image_size)
    # The style prediction model was trained with image size 256 and it's the 
    # recommended image size for the style image (though, other sizes work as 
    # well but will lead to different results).
    style_img_size = (256, 256)  # Recommended to keep it at 256.

    content_image = load_image(content_image_url, content_img_size)
    style_image = load_image(style_image_url, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
    
    # Load TF Hub module.

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    outputs = hub_module(content_image, style_image)
    stylized_image = outputs[0]

    # Stylize content image with given style image.
    # This is pretty fast within a few milliseconds on a GPU.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    # Get the base64-encoded image string
    encoded_image = show_and_return_base64([stylized_image],
                                       titles=['Stylized image'])
    return render_template('index.html', encoded_image=encoded_image)
    
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

  # @title Load example images  { display-mode: "form" }

# content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
# style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg'  # @param {type:"string"}
# output_image_size = 384  # @param {type:"integer"}

# # The content image size can be arbitrary.
# content_img_size = (output_image_size, output_image_size)
# # The style prediction model was trained with image size 256 and it's the 
# # recommended image size for the style image (though, other sizes work as 
# # well but will lead to different results).
# style_img_size = (256, 256)  # Recommended to keep it at 256.

# content_image = load_image(content_image_url, content_img_size)
# style_image = load_image(style_image_url, style_img_size)
# stylized_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
# print(stylized_image, "ll")
# # show_n([content_image, style_image], ['Content image', 'Style image'])
# # Get the base64-encoded image string
# encoded_image = show_and_return_base64([content_image, style_image, stylized_image],
#                                        titles=['Original content image', 'Style image', 'Stylized image'])
# print(encoded_image, "enc")



# Load TF Hub module.

# hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# hub_module = hub.load(hub_handle)

# outputs = hub_module(content_image, style_image)
# stylized_image = outputs[0]

# Stylize content image with given style image.
# This is pretty fast within a few milliseconds on a GPU.

# outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
# stylized_image = outputs[0]

# Visualize input images and the generated stylized image.

# show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])

if __name__ == '__main__':
    app.run(debug=True,port=3000)



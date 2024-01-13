import pandas as pd
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify, render_template, request

nltk.download('vader_lexicon')

app = Flask(__name__, template_folder='')

# Read the dataset from a CSV file
dataset_path = 'Songs.csv'
df = pd.read_csv(dataset_path)

sia = SentimentIntensityAnalyzer()

def get_emotion_from_lyrics(lyrics):
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
    
def search_song(song_name):
    result = df[df['Title'].str.lower() == song_name.lower()]

    if not result.empty:
        polarity_score = sia.polarity_scores(result['Lyrics'].values[0])['compound']
        emotion = get_emotion_from_lyrics(result['Lyrics'].values[0])

        return {
            'Title': result['Title'].values[0],
            'Emotion': emotion
        }
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/get_emotions')
def get_emotions():

    # Add a new column 'Genre' to store the predicted genre based on polarity
    df['Emotion'] = df['Lyrics'].apply(get_emotion_from_lyrics)

    # Prepare the result as JSON
    result = df[['Title', 'Emotion']].to_dict(orient='records')

    return jsonify(result)

@app.route('/search')
def search():
    # Get the user-inputted song name from the query parameters
    user_input_song = request.args.get('song_name', '')

    # Perform the search and get the result
    search_result = search_song(user_input_song)

    if search_result:
        return jsonify(search_result)
    else:
        return jsonify({'error': 'Song not found'})

if __name__ == '__main__':
    app.run(debug=True)

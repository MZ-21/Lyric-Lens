import pandas as pd
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify, render_template

nltk.download('vader_lexicon')

app = Flask(__name__)

def get_emotion_from_lyrics(lyrics):
    sia = SentimentIntensityAnalyzer()
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_genres')
def get_emotions():
    # Read the dataset from a CSV file
    dataset_path = 'Songs.csv'
    df = pd.read_csv(dataset_path)

    # Create a SentimentIntensityAnalyzer instance
    sia = SentimentIntensityAnalyzer()

    # Calculate polarity scores for each song's lyrics
    df['Polarity'] = df['Lyrics'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Add a new column 'Genre' to store the predicted genre based on polarity
    df['Genre'] = df['Polarity'].apply(get_emotion_from_lyrics)

    # Prepare the result as JSON
    result = df[['Title', 'Genre']].to_dict(orient='records')

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

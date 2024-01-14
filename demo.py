import pandas as pd
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, jsonify, render_template, request

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
    return render_template('index.html')
    
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

if __name__ == '__main__':
    app.run(debug=True)

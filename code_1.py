import pandas as pd
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

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

# Read the dataset from a CSV file
dataset_path = 'Songs.csv'
df = pd.read_csv(dataset_path)

# Set the option to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Add a new column 'Emotion' to store the predicted emotion
df['Genre'] = df['Lyrics'].apply(get_emotion_from_lyrics)

# Display the result
print(df[['Title', 'Genre']])

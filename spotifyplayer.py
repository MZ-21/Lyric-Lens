
import requests
from pygame import mixer
from flask import Flask, jsonify, render_template, request
app = Flask(__name__, template_folder='')

#Instantiate mixer
mixer.init()

# Replace 'YOUR_CLIENT_ID' and 'YOUR_CLIENT_SECRET' with your Spotify API credentials
client_id = 'ba032a5182044df09ddb3a5128f0c848'
client_secret = 'df162ca97a6049a4bfe20bab8b877463'

# Replace 'YOUR_TRACK_NAME' with the name of the track you want to play
track_name = 'me'

# Step 1: Authenticate with Spotify API and get an access token
auth_url = 'https://accounts.spotify.com/api/token'
auth_data = {'grant_type': 'client_credentials'}
auth_response = requests.post(auth_url, auth=(client_id, client_secret), data=auth_data)
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
    with open(output_file, 'wb') as f:
        f.write(response.content)

# Replace 'your_url_here' with the actual URL of the MP3 file
mp3_url = preview_url
output_filename = track_name+".mp3"
mixer.music.load(output_filename)

# Actions to control the music
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

# Load audio file
@app.route('/')
def index():
    mixer.music.play(1)
    return render_template('music.html', track_name=track_name)

if __name__ == '__main__':
    app.run(debug=True)



download_mp3(mp3_url, output_filename)

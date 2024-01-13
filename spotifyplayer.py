# import spotipy
# import pygame
# from spotipy.oauth2 import SpotifyClientCredentials
# from pygame import mixer


# # Replace 'YOUR_CLIENT_ID' and 'YOUR_CLIENT_SECRET' with your actual credentials
# client_credentials_manager = SpotifyClientCredentials(client_id='ba032a5182044df09ddb3a5128f0c848', client_secret='df162ca97a6049a4bfe20bab8b877463')
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# # Replace 'SONG_NAME' with the name of the song you want to search for
# results = sp.search(q='Sorry', type='track', limit=1)
# if results['tracks']['items']:
#     print(results, " results")
#     track_id = results['tracks']['href']
#     print(track_id)
#     mixer.init()
#     mixer.music.load(track_id)
#     mixer.music.play()
#     # Add a delay or loop to keep the program running while the music plays
#     pygame.time.wait(5000)      


# print(results)


# import vlc
import requests
import pygame

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
print(search_response_data, "res")

# Step 3: Get the preview URL
if 'tracks' in search_response_data and 'items' in search_response_data['tracks'] and search_response_data['tracks']['items']:
    track = search_response_data['tracks']['items'][0]
    preview_url = track.get('preview_url')

    if preview_url:
        print(preview_url, ' iiiii')
        # Create a VLC media instance
        # media = vlc.Media(preview_url)
        
        # # Create a media player instance
        # player = vlc.MediaPlayer(media)
        
        # # Play the media
        # player.play()
        
        # # Wait for the playback to finish (adjust as needed)
        # player.get_media().get_mrl()
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

download_mp3(mp3_url, output_filename)

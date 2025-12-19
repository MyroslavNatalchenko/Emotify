import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth
from datetime import datetime, timedelta, timezone

CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
REDIRECT_URI = 'http://127.0.0.1:8888/callback'

scope = "user-read-recently-played"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=scope))

def get_tracks_last_hour():
    results = sp.current_user_recently_played(limit=50)

    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

    found_tracks = []

    for item in results['items']:
        played_at_str = item['played_at'].replace('Z', '+00:00')
        played_at = datetime.fromisoformat(played_at_str)

        if played_at > one_hour_ago:
            track = item['track']
            track_name = track['name']
            track_id = track['id']
            artist_name = track['artists'][0]['name']
            found_tracks.append(f"{artist_name} - {track_name} - {track_id}")

    return found_tracks


if __name__ == "__main__":
    print("Last played songs during last hour:")
    tracks = get_tracks_last_hour()

    if tracks:
        for t in tracks:
            print(t)
    else:
        print("No songs played")
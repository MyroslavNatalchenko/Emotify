import os
import requests
from flask import Flask, render_template, request, redirect, session, url_for
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())
app.secret_key = os.urandom(24)

# Konfiguracja
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:8000')
BACKEND_TIMEOUT = 30

sp_oauth = SpotifyOAuth(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
    redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI'),
    scope='user-library-read user-top-read user-read-recently-played'
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        token_info = sp_oauth.get_access_token(code, as_dict=True)
    session['token_info'] = token_info
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    token_info = session.get('token_info')
    if not token_info:
        return redirect('/login')

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user = sp.current_user()

    recent = sp.current_user_recently_played(limit=10)
    tracks = []
    for item in recent['items']:
        tracks.append({
            'id': item['track']['id'],
            'name': item['track']['name'],
            'artist': item['track']['artists'][0]['name'],
            'image': item['track']['album']['images'][0]['url']
        })

    return render_template('dashboard.html', user=user, tracks=tracks)


@app.route('/analyze/<track_id>')
def analyze(track_id):
    token_info = session.get('token_info')
    if not token_info:
        return redirect('/login')

    sp = spotipy.Spotify(auth=token_info['access_token'])
    track = sp.track(track_id)

    emotions = []
    is_mock = True
    error_message = None

    try:
        response = requests.post(
            f'{BACKEND_API_URL}/predict',
            json={
                'track_id': track_id,
                'preview_url': track.get('preview_url'),
                'track_name': track['name'],
                'artist_name': track['artists'][0]['name']
            },
            timeout=BACKEND_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            emotions = data.get('emotions', [])
            is_mock = False
        else:
            error_message = f"Backend error: {response.status_code}"
            emotions = _get_mock_emotions()

    except requests.exceptions.ConnectionError:
        error_message = "Backend API not running (connection refused)"
        emotions = _get_mock_emotions()

    except requests.exceptions.Timeout:
        error_message = f"Backend timeout (>{BACKEND_TIMEOUT}s)"
        emotions = _get_mock_emotions()

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        emotions = _get_mock_emotions()

    return render_template('result.html',
                           track=track,
                           emotions=emotions,
                           is_mock=is_mock,
                           error_message=error_message)


def _get_mock_emotions():
    # Mock data dla testowania UI
    return [
        ('happy', 0.85),
        ('energetic', 0.72),
        ('uplifting', 0.68),
        ('party', 0.55),
        ('melodic', 0.48),
        ('danceable', 0.42),
        ('positive', 0.38),
        ('fun', 0.35),
        ('emotional', 0.30),
        ('relaxing', 0.25)
    ]


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

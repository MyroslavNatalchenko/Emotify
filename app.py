import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth
from model_usage.predict import EmotionPredictor

load_dotenv()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model/trained_model/audio_rec_model.pth'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

predictor = EmotionPredictor(MODEL_PATH)

sp_oauth = SpotifyOAuth(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
    redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI'),
    scope='user-library-read user-top-read user-read-recently-played',
    cache_handler=None
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model_loaded": predictor.model is not None})


@app.route('/api/auth/login')
def api_login():
    auth_url = sp_oauth.get_authorize_url()
    return jsonify({"url": auth_url})


@app.route('/api/auth/exchange', methods=['POST'])
def api_exchange_token():
    data = request.json
    code = data.get('code')
    if not code:
        return jsonify({"error": "No code provided"}), 400
    try:
        token_info = sp_oauth.get_access_token(code)

        if not token_info or 'access_token' not in token_info:
            return jsonify({"error": "Invalid token response"}), 400

        return jsonify(token_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/analyze/upload', methods=['POST'])
@limiter.limit("5 per minute")
def analyze_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Wrong file format (use: mp3, wav, ogg)"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(filepath)
        emotions = predictor.predict(filepath)
        return jsonify({"emotions": emotions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
         if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/analyze/spotify', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_spotify():
    data = request.json
    preview_url = data.get('preview_url')
    track_id = data.get('track_id')

    if not preview_url:
        return jsonify({"error": "No preview URL available for this track"}), 400

    filename = f"spotify_{track_id}.mp3"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        doc = requests.get(preview_url, timeout=10)

        if doc.status_code != 200:
            return jsonify({"error": "Could not download preview from Spotify (Invalid Status)"}), 400

        with open(filepath, 'wb') as f:
            f.write(doc.content)

        emotions = predictor.predict(filepath)
        return jsonify({"emotions": emotions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large (Max 16MB)"}), 413

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
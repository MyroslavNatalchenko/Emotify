import os
import re
import shutil
import time
import urllib.request
from urllib.parse import quote_plus
import requests
import yt_dlp
from mutagen.easyid3 import EasyID3
from mutagen.id3 import APIC, ID3
from rich.console import Console
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

DOWNLOAD_DIR = "../music"
TEMP_DIR = os.path.join(DOWNLOAD_DIR, "tmp")

SPOTIPY_CLIENT_ID = os.environ.get("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.environ.get("SPOTIPY_CLIENT_SECRET")

client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

console = Console()
file_exists_action = ""

def sanitize_filename(name: str) -> str:
    """
    Removes illegal characters from filenames to ensure filesystem compatibility.
    """
    return re.sub(r'[\/\\\|\?\*\:\>\<"]', '', name)


def validate_url(sp_url: str) -> str:
    """
    Validates if the provided string is a properly formatted Spotify URL.
    """
    if re.search(r"^(https?://)?open\.spotify\.com/(playlist|track)/.+$", sp_url):
        return sp_url
    raise ValueError("Invalid Spotify URL")


def prompt_exists_action() -> bool:
    """
    Asks the user how to handle files that already exist (Skip, Replace, etc.).
    """
    global file_exists_action
    if file_exists_action == "SA":  # SA == 'Skip All'
        return False
    elif file_exists_action == "RA":  # RA == 'Replace All'
        return True

    print("This file already exists.")
    while True:
        resp = (
            input("replace[R] | replace all[RA] | skip[S] | skip all[SA]: ")
            .upper()
            .strip()
        )
        if resp in ("RA", "SA"):
            file_exists_action = resp
        if resp in ("R", "RA"):
            return True
        elif resp in ("S", "SA"):
            return False
        print("---Invalid response---")

def get_track_info(track_url: str) -> dict:
    """
    Fetches metadata (artist, title, album, cover art) for a specific Spotify track.
    """
    try:
        res = requests.get(track_url)
        if res.status_code != 200:
            raise ValueError("Invalid Spotify track URL")
    except Exception:
        pass

    track = sp.track(track_url)

    track_metadata = {
        "artist_name": track["artists"][0]["name"],
        "track_title": track["name"],
        "track_number": track["track_number"],
        "isrc": track["external_ids"].get("isrc", ""),
        "album_art": track["album"]["images"][1]["url"],
        "album_name": track["album"]["name"],
        "release_date": track["album"]["release_date"],
        "artists": [artist["name"] for artist in track["artists"]],
    }

    return track_metadata


def get_playlist_info(sp_playlist: str) -> list:
    """
    Fetches metadata for all tracks contained in a public Spotify playlist.
    """
    res = requests.get(sp_playlist)
    if res.status_code != 200:
        raise ValueError("Invalid Spotify playlist URL")

    pl = sp.playlist(sp_playlist)
    if not pl["public"]:
        raise ValueError(
            "Can't download private playlists. Change your playlist's state to public."
        )

    playlist = sp.playlist_tracks(sp_playlist)
    tracks = [item["track"] for item in playlist["items"]]
    tracks_info = []

    for track in tracks:
        if track:
            track_url = track['external_urls'].get('spotify', f"https://open.spotify.com/track/{track['id']}")
            track_info = get_track_info(track_url)
            tracks_info.append(track_info)

    return tracks_info

def find_youtube(query: str) -> str:
    """
    Searches YouTube for the provided query string and returns the URL of the first video result.
    """
    encoded_query = quote_plus(query)
    search_link = f"https://www.youtube.com/results?search_query={encoded_query}"

    count = 0
    response = None
    while count < 3:
        try:
            response = urllib.request.urlopen(search_link)
            break
        except:
            count += 1

    if response is None:
        raise ValueError("Please check your internet connection and try again later.")

    html = response.read().decode("utf-8")
    search_results = re.findall(r"watch\?v=(\S{11})", html)

    if not search_results:
        raise ValueError("No YouTube results found")

    return "https://www.youtube.com/watch?v=" + search_results[0]


def download_yt(yt_link: str) -> str:
    """
    Downloads the audio from a YouTube video, converts it to MP3, and saves it to a temporary location.
    """
    out_template = os.path.join(TEMP_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "js_runtime": "node",
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar", "44100",  # sample rate 44.1 kHz
            "-ac", "1",  # mono (1 channel)
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_link, download=True)
        title = sanitize_filename(info["title"])
        return os.path.join(TEMP_DIR, f"{title}.mp3")


def set_metadata(metadata: dict, file_path: str):
    """
    Embeds ID3 metadata (tags and album art) into the downloaded MP3 file.
    """
    mp3file = EasyID3(file_path)

    # add text metadata
    mp3file["albumartist"] = metadata["artist_name"]
    mp3file["artist"] = metadata["artists"]
    mp3file["album"] = metadata["album_name"]
    mp3file["title"] = metadata["track_title"]
    mp3file["date"] = metadata["release_date"]
    mp3file["tracknumber"] = str(metadata["track_number"])
    if metadata["isrc"]:
        mp3file["isrc"] = metadata["isrc"]
    mp3file.save()

    # add album cover
    audio = ID3(file_path)
    with urllib.request.urlopen(metadata["album_art"]) as albumart:
        audio["APIC"] = APIC(
            encoding=3, mime="image/jpeg", type=3, desc="Cover", data=albumart.read()
        )
    audio.save(v2_version=3)

def main():
    """
    Main orchestrator: handles user input, iterates through tracks, downloads audio, and moves files.
    """
    try:
        url = validate_url(input("Enter a spotify url: ").strip())
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    songs = []
    if "track" in url:
        songs = [get_track_info(url)]
    elif "playlist" in url:
        songs = get_playlist_info(url)

    start = time.time()
    downloaded = 0

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for i, track_info in enumerate(songs, start=1):
        search_term = f"{track_info['artist_name']} {track_info['track_title']} audio"

        try:
            video_link = find_youtube(search_term)
        except ValueError as e:
            console.print(f"[red]Error finding video: {e}[/red]")
            continue

        console.print(
            f"[magenta]({i}/{len(songs)})[/magenta] Downloading '[cyan]{track_info['artist_name']} - {track_info['track_title']}[/cyan]'..."
        )

        try:
            audio = download_yt(video_link)
            if audio:
                set_metadata(track_info, audio)

                final_filename = os.path.basename(audio)
                final_path = os.path.join(DOWNLOAD_DIR, final_filename)

                if os.path.exists(final_path):
                    if prompt_exists_action():
                        os.replace(audio, final_path)
                    else:
                        os.remove(audio)
                else:
                    os.replace(audio, final_path)

                console.print(
                    "[blue]______________________________________________________________________"
                )
                downloaded += 1
            else:
                print("Download failed or file exists. Skipping...")
        except Exception as e:
            console.print(f"[red]Error processing track: {e}[/red]")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    end = time.time()
    print()

    print(f"Download location: {os.path.abspath(DOWNLOAD_DIR)}")
    console.print(
        f"DOWNLOAD COMPLETED: {downloaded}/{len(songs)} song(s) dowloaded".center(
            70, " "
        ),
        style="on green",
    )
    console.print(
        f"Total time taken: {round(end - start)} sec".center(70, " "), style="on white"
    )

    try:
        sp._session.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
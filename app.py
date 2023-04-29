import sounddevice as sd
import scipy.io.wavfile as wavfile

from flask import Flask, request
from pathlib import Path
import json

from models import Event
from whisper_utils.whisper_utils import call_whisper, create_audio_event_matching, DataClassEncoder

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

audio_frames = []
audio_samplerate = 44100
audio_channels = 1
audio_duration = 5


def start_recording():
    global audio_frames
    audio_frames = sd.rec(int(audio_samplerate * audio_duration), samplerate=audio_samplerate, channels=audio_channels)


def stop_recording():
    global audio_frames
    cwd = Path.cwd()

    wavfile.write(cwd / "audio.wav", audio_samplerate, audio_frames)


@app.route("/start_recording", methods=["GET"])
def start_recording_route():
    start_recording()
    return "Recording started", 200


@app.route("/stop_recording", methods=["GET"])
def stop_recording_route():
    stop_recording()
    return "Recording stopped", 200


@app.route('/api/audio', methods=['GET'])
def transcribe_audio():
    cwd = Path.cwd()

    whisper_result = call_whisper(cwd / 'audio.wav')

    event_json = json.load(open(cwd / 'events.json', 'r'))
    events = [Event(event=json_event['event'], name=json_event['name'], time=json_event['time'], txt=json_event['txt']) for json_event in
              event_json]

    event_matchings = create_audio_event_matching(whisper_result, events)
    json_data = json.dumps(event_matchings, cls=DataClassEncoder)

    with open(cwd / 'event_matching.json', 'w') as f:
        json.dump(json_data, f)

    return json_data, 200


@app.route('/api/events', methods=['PUT'])
def save_events():
    if not request.is_json:
        return 'No JSON data provided', 400
    data = request.get_json()
    cwd = Path.cwd()
    with open(cwd / 'events.json', 'w') as f:
        json.dump(data, f)
    return 'JSON data saved successfully', 200


@app.route('/api/events', methods=['GET'])
def get_events():
    cwd = Path.cwd()
    with open(cwd / 'event_matching.json', 'r') as f:
        data = json.load(f)
    return data, 200


if __name__ == '__main__':
    app.run()

import dataclasses

from flask import Flask, request
from pathlib import Path
import json

from models import Event
from whisper_utils.whisper_utils import call_whisper, create_audio_event_matching, DataClassEncoder

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB


@app.route('/api/audio', methods=['PUT'])
def transcribe_audio():
    if 'file' not in request.files:
        return 'No file provided', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    cwd = Path.cwd()
    file.save(cwd / 'audio.wav')

    whisper_result = call_whisper('audio.wav')

    event_json = json.load(open(cwd / 'events.json', 'r'))
    events = [Event(event=json_event['event'], name=json_event['name'], time=json_event['time'], txt=json_event['txt']) for json_event in
              event_json]

    event_matchings = create_audio_event_matching(whisper_result, events)
    json_data = json.dumps(event_matchings, cls=DataClassEncoder)

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


if __name__ == '__main__':
    app.run()

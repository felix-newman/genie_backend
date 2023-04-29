import sys
from pathlib import Path
from typing import Dict, Any, List

import whisper_timestamped as whisper
import json

from models import Event, EventMatching
from collections import defaultdict


def call_whisper(audio_file):
    audio = whisper.load_audio(audio_file)
    print(audio, file=sys.stdout)
    model = whisper.load_model("tiny.en", device="cpu")
    return whisper.transcribe(model, audio, language="en")


def create_audio_event_matching(whisper_result: Dict[str, Any], events: List[Event]) -> List[EventMatching]:
    segments = whisper_result["segments"]
    matches = defaultdict(list)

    for event in events:
        matching_idx = 0

        for idx, segment in enumerate(segments):
            if segment['start'] < event.time:
                matching_idx = idx
            else:
                break

        matches[matching_idx].append(event)

    transformed_matches = [EventMatching(text=segment['text'], events=matches[segment_idx]) for segment_idx, segment in enumerate(segments)]
    return transformed_matches


class DataClassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Event, EventMatching)):
            return obj.__dict__
        return super().default(obj)


if __name__ == '__main__':
    cwd = Path.cwd()
    event_json = json.load(open(cwd / 'events.json', 'r'))
    events = [Event(event=json_event['event'], name=json_event['name'], time=json_event['time'], txt=json_event['txt']) for json_event in
              event_json]
    whisper_result = call_whisper('audio.wav')

    event_matchings = create_audio_event_matching(whisper_result, events)
    json_data = json.dumps(event_matchings, cls=DataClassEncoder)

    print(json_data)

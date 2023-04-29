from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    event: str = ""
    id: str = " "
    txt: str = " "
    html_class: str = " "
    name: str = ""
    alt: str = ""
    tag: str = ""
    key: str = ""
    url: str = ""
    time: float = 0.0


@dataclass
class EventMatching:
    text: str
    events: List[Event]

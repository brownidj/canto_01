from dataclasses import dataclass
import json, os

CONFIG_FILE = "config.json"

@dataclass
class Preferences:
    play_mode: str        # "Pronunciation" | "Listen & Choose" | "Future Option"
    jyutping_mode: str    # "Learner" | "Strict" | "Borders"
    tts_rate: int         # e.g. 180

def load_prefs(defaults: Preferences) -> Preferences:
    if not os.path.exists(CONFIG_FILE):
        return defaults
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return defaults
    return Preferences(
        play_mode=data.get("CURRENT_PLAY_MODE", defaults.play_mode),
        jyutping_mode=data.get("CURRENT_JYUTPING_MODE", defaults.jyutping_mode),
        tts_rate=int(data.get("TTS_RATE", defaults.tts_rate)),
    )

def save_prefs(p: Preferences) -> None:
    data = {
        "CURRENT_PLAY_MODE": p.play_mode,
        "CURRENT_JYUTPING_MODE": p.jyutping_mode,
        "TTS_RATE": int(p.tts_rate),
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
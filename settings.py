# settings.py
# Things that can be changed

SPEAK_ON_CLICK = True   # whether to auto-speak when a tile is clicked
TTS_RATE_DEFAULT = 180          # macOS 'say' words per minute (100–260 typical)
NUMBER_OF_CHANCES = 3   # number of attempts to click a tile before giving up
TRICKY_INITIAL_EXPLANATIONS = True
DEBUG = True
PLAY_MODE_DEFAULT = "Pronunciation"  # default play mode ("Pronunciation" | "Listen & Choose" | "Future Option")
CURRENT_PLAY_MODE = PLAY_MODE_DEFAULT


JYUTPING_MODE_DEFAULT = "strict"  # "learner" | "strict" | "strict_with_word_boundaries"
CURRENT_JYUTPING_MODE = JYUTPING_MODE_DEFAULT
JYUTPING_WORD_BOUNDARY_MARKER = " · "

SWATCH_BASELINE_OFFSET_PX = 0  # try 6–14 until it visually matches your Jyutping baseline
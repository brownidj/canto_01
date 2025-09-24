# constants.py
# Central knobs you can tweak without touching the app code.

TONE_DESCRIPTIONS = {
    "1": "High level",
    "2": "High rising",
    "3": "Mid level",
    "4": "Low falling",
    "5": "Low rising",
    "6": "Low level",
}

# ---------------- Tone colors (pastel-ish) ----------------
# Keep these hex values as you like; they’ll be used everywhere in the app.
TONE_COLOURS = {
    "1": "#9CC5D6",  # High level: slightly duller sky blue
    "2": "#8AD6F0",  # High rising: marginally brighter blue
    "3": "#C8A2C8",  # Mid level: lavender
    "4": "#FFB347",  # Low falling: deep amber
    "5": "#FFE1B2",  # Low rising: lighter peach
    "6": "#E3C9A6",  # Low level: soft tan (slightly duller)
}

# ---------------- “Both” mode ratio ----------------
# Portion of characters vs words when building the “Both” pool.
# Example: 0.6 means 40% characters, 60% words.
BOTH_CHAR_RATIO = 0.40

TOOLTIP_DELAY = 2000  # Delay in milliseconds before showing tooltips.

# ---------------- Tooltip Texts ---------------- #
PLAY_TOOLTIP = "Play a random tile’s pronunciation."
PLAYMODE_TOOLTIP = "Choose how to practice: Listen & Choose or Hear Pronunciation."
MODE_TOOLTIP = "Pick which list the tiles come from."
TOPN_TOOLTIP = "Set how many frequent items to include."
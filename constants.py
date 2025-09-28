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
    "1": "#2E89E6",  # High level: slightly darker blue for distinction from #33CCFF
    "2": "#33CCFF",  # High rising: bright sky-blue/teal (improved visibility)
    "3": "#9C27B0",  # Mid level: lighter purple (better visibility against light grey background)
    "4": "#E65100",  # Low falling: vivid orange (more saturated/darker for contrast)
    "5": "#FF8F00",  # Low rising: bright amber (more saturated for contrast)
    "6": "#A1887F",  # Low level: lighter brown (lighter for better contrast)
}

TONE_KEY_TEXT_COLOURS = {
    "1": "#FFFFFF",
    "2": "#FFFFFF",
    "3": "#FFFFFF",
    "4": "#FFFFFF",
    "5": "#FFFFFF",
    "6": "#FFFFFF",
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

# Centralized Jyutping style ↔ label mappings
STYLE_TO_LABEL = {
    "learner": "Learner",
    "strict": "Strict",
    "strict_with_word_boundaries": "Borders",
}
LABEL_TO_STYLE = {v: k for k, v in STYLE_TO_LABEL.items()}

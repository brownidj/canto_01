# constants.py
# Central knobs you can tweak without touching the app code.

# ---------------- Tone colors (pastel-ish) ----------------
# Keep these hex values as you like; they’ll be used everywhere in the app.
TONE_COLORS = {
    "high_level": "#cce6ff",    # Tone 1  (high level)  – slightly duller light blue
    "high_rising": "#bfe6ff",   # Tone 2  (high rising) – marginally brighter blue (per your earlier ask)
    "mid_level": "#c6f7c3",     # Tone 3  (mid level)   – light green
    "low_falling": "#ffe6b8",   # Tone 4  (low falling) – light orange/amber
    "low_rising": "#ffd9b3",    # Tone 5  (low rising)  – a bit brighter than 6 (per earlier tuning)
    "low_level": "#ffd1a6",     # Tone 6  (low level)   – slightly duller than 5 (per earlier tuning)
}

# ---------------- “Both” mode ratio ----------------
# Portion of characters vs words when building the “Both” pool.
# Example: 0.6 means 40% characters, 60% words.
BOTH_CHAR_RATIO = 0.40
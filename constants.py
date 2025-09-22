# constants.py
# Central knobs you can tweak without touching the app code.

# ---------------- Tone colors (pastel-ish) ----------------
# Keep these hex values as you like; they’ll be used everywhere in the app.
TONE_COLORS = {
    1: "#cce6ff",  # Tone 1  (high level)  – slightly duller light blue
    2: "#bfe6ff",  # Tone 2  (high rising) – marginally brighter blue (per your earlier ask)
    3: "#c6f7c3",  # Tone 3  (mid level)   – light green
    4: "#ffe6b8",  # Tone 4  (low falling) – light orange/amber
    5: "#ffd9b3",  # Tone 5  (low rising)  – a bit brighter than 6 (per earlier tuning)
    6: "#ffd1a6",  # Tone 6  (low level)   – slightly duller than 5 (per earlier tuning)
}

# ---------------- “Both” mode ratio ----------------
# Portion of characters vs words when building the “Both” pool.
# Example: 0.6 means 40% characters, 60% words.
BOTH_CHAR_RATIO = 0.
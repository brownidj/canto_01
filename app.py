# -*- coding: utf-8 -*-
import re
import os
import random
import collections
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import warnings
# Audio helpers for TTS (non-blocking)
import platform
import subprocess
import threading
# Silence noisy UserWarnings emitted by wordseg/pkg_resources during import
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)
import pycantonese

APP_TITLE = "Cantonese (HKCanCor) – 1×5 with Meanings"
CJK_RE = re.compile(u"[\u4E00-\u9FFF]+")
DICT_FILENAME = os.path.join("assets", "cedict_ts.u8")  # CC-CEDICT in assets/
CC_CANTO_FILENAME = os.path.join("assets", "cc_canto.u8")  # CC-Canto in assets/


# A tiny built-in fallback glossary for very common forms (traditional)
# (Only used when CC-CEDICT file isn't present or a given key isn't found.)
MINI_GLOSS = {
    u"我": ["I", "me"],
    u"你": ["you"],
    u"佢": ["he", "she", "they"],
    u"唔": ["not (Cantonese)"],
    u"有": ["to have", "there is"],
    u"冇": ["not have (Cantonese)"],
    u"去": ["to go"],
    u"食": ["to eat"],
    u"飲": ["to drink"],
    u"學": ["to learn; school (in compounds)"],
    u"講": ["to speak; to say"],
    u"睇": ["to look; to watch (Cantonese)"],
    u"聽": ["to listen"],
    u"車": ["vehicle; car"],
    u"書": ["book"],
    u"電": ["electric; electricity"],
    u"心": ["heart; mind"],
    u"頭": ["head; top"],
    u"手": ["hand"],
    u"腳": ["leg; foot"],
    u"日": ["sun; day"],
    u"月": ["moon; month"],
    u"年": ["year"],
    u"香港": ["Hong Kong"],
    u"廣東話": ["Cantonese (language)"],
    u"你好": ["hello"],
    u"然": ["thus", "so", "like that"],
    u"其": ["its", "his", "her", "their"],
    u"之": ["(classical genitive/linker)"],
    u"以": ["to use", "by means of"],
    u"於": ["at", "in", "to"],
    u"而": ["and", "and then", "but"],
    u"則": ["then", "in that case"],
    u"不": ["not"],
    u"也": ["also", "too"],
    u"的": ["(structural/possessive particle)"],
    u"了": ["(aspect particle)", "completed action"],
    u"嗎": ["(question particle)"],
    u"呢": ["(particle: how about…; continuative)"],
    u"吧": ["(particle: suggestion/softener)"],
    u"個": ["classifier (general)"],
    u"嘢": ["thing; stuff", "food/drink (in collocations like 食嘢、飲嘢)"],
    u"食嘢": ["to eat (colloquial)", "to get something to eat"],
    u"飲嘢": ["to drink (colloquial)", "to get something to drink"]
}

# ------------------------ Dictionary (CC-CEDICT) ------------------------ #

def load_cedict_dict(path):
    """
    Load a subset of CC-CEDICT into a dictionary: {traditional: [glosses...]}
    File format lines look like:
        傳統 簡体 [pin1 yin1] /meaning 1/meaning 2/...
    Returns a dict. If file not present, returns {}.
    """
    if not os.path.exists(path):
        return {}

    gloss = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Split: Traditional, Simplified, rest
                # Find first space (traditional) and second space (simplified)
                parts = line.split(" ", 2)
                if len(parts) < 3:
                    continue
                trad = parts[0]
                rest = parts[2]
                # meanings separated by /.../ after the pinyin bracket block
                slash_idx = rest.find(" /")
                if slash_idx == -1:
                    continue
                meanings_part = rest[slash_idx + 1:]  # starts with /meaning...
                # strip leading/trailing slashes and split
                meanings = [m for m in meanings_part.strip("/").split("/") if m]
                if trad and meanings:
                    # Append to list if multiple entries per headword
                    existing = gloss.get(trad, [])
                    # Avoid duplicates
                    for m in meanings:
                        if m not in existing:
                            existing.append(m)
                    gloss[trad] = existing
    except Exception:
        # On any parsing error, just return what we have so far (or empty)
        pass
    return gloss


def load_cc_canto_dict(path):
    """
    Load CC-Canto (Cantonese-focused CEDICT variant) into a dict {traditional: [glosses...]}.
    Format is typically the same as CEDICT (traditional simplified [jyutping/... or pinyin] /.../).
    Returns {} if file not present.
    """
    if not os.path.exists(path):
        return {}
    gloss = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ", 2)
                if len(parts) < 3:
                    continue
                trad = parts[0]
                rest = parts[2]
                slash_idx = rest.find(" /")
                if slash_idx == -1:
                    continue
                meanings_part = rest[slash_idx + 1:]
                meanings = [m for m in meanings_part.strip("/").split("/") if m]
                if trad and meanings:
                    existing = gloss.get(trad, [])
                    for m in meanings:
                        if m not in existing:
                            existing.append(m)
                    gloss[trad] = existing
    except Exception:
        pass
    return gloss

# ------------------ Merged dictionary lookup (CC-Canto > CEDICT > MINI_GLOSS > char comp) ------------------ #
def lookup_meaning_merged(word, cc_canto_dict, cedict_dict):
    """
    Prefer CC-Canto, then CC-CEDICT, then MINI_GLOSS.
    If still missing and multi-character, compose from individual characters.
    Returns a list of gloss strings.
    """
    # Direct matches (priority order)
    if word in cc_canto_dict:
        return cc_canto_dict[word]
    if word in cedict_dict:
        return cedict_dict[word]
    if word in MINI_GLOSS:
        return MINI_GLOSS[word]

    # Compose from characters if possible
    if len(word) > 1:
        parts = []
        for ch in word:
            if ch in cc_canto_dict:
                parts.append(ch + ": " + "; ".join(cc_canto_dict[ch][:2]))
            elif ch in cedict_dict:
                parts.append(ch + ": " + "; ".join(cedict_dict[ch][:2]))
            elif ch in MINI_GLOSS:
                parts.append(ch + ": " + "; ".join(MINI_GLOSS[ch]))
        if parts:
            return [" + ".join(parts)]

    return ["(meaning not available)"]


def lookup_meaning(word, cedict):
    """
    Look up meaning(s) for a word from cedict (traditional).
    Falls back to MINI_GLOSS; if still missing, returns a placeholder.
    """
    # Exact match first
    if word in cedict:
        return cedict[word]
    if word in MINI_GLOSS:
        return MINI_GLOSS[word]

    # If multi-character, try to compose meanings from individual chars
    if len(word) > 1:
        parts = []
        for ch in word:
            if ch in cedict:
                parts.append(ch + ": " + "; ".join(cedict[ch][:2]))
            elif ch in MINI_GLOSS:
                parts.append(ch + ": " + "; ".join(MINI_GLOSS[ch]))
        if parts:
            return [" + ".join(parts)]

    return ["(meaning not available)"]


# --------------------------- Corpus & Frequency -------------------------- #

def load_hkcancor_tokens():
    """
    Load HKCanCor and return a list of token strings.
    We try several APIs for robustness across pycantonese versions.
    """
    corpus = pycantonese.hkcancor()

    # 1) words()
    try:
        words = corpus.words()
        if words:
            return words
    except Exception:
        pass

    # 2) sents() → list[list[str]]
    try:
        sents = corpus.sents()
        tokens = []
        for s in sents:
            for w in s:
                tokens.append(w)
        if tokens:
            return tokens
    except Exception:
        pass

    # 3) utterances() fallback
    tokens = []
    try:
        for utt in corpus.utterances():
            ws = getattr(utt, "words", None)
            if ws:
                for w in ws:
                    tokens.append(w)
            else:
                text = getattr(utt, "transcript_text", "")
                if text:
                    tokens.append(text)
    except Exception:
        pass

    return tokens


def _norm_jyut(j):
    """
    Normalize various Jyutping return shapes to a plain string.
    Accepts strings, tuples like ('對', 'deoi3'), or lists like ['deoi3','deoi6'].
    Preference order: the second element if tuple of (hanzi, jyut), else first string found.
    """
    if j is None:
        return ""
    if isinstance(j, str):
        return j
    if isinstance(j, tuple):
        if len(j) >= 2 and isinstance(j[1], str):
            return j[1]
        for x in j:
            if isinstance(x, str):
                return x
        return ""
    if isinstance(j, list):
        for x in j:
            if isinstance(x, str) and x:
                return x
        return ""
    return str(j)


# ---------------------- English Approximation ---------------------- #

_JP_APPROX_MAP = {
    "aa": "ah", "aai": "eye", "aau": "ow",
    "ai": "eye", "au": "ow",
    "e": "eh", "ei": "ay", "eu": "eh-oo",
    "eoi": "oey", "eo": "er", "oe": "ur",
    "i": "ee", "iu": "yoo",
    "o": "aw", "oi": "oy", "ou": "oh",
    "u": "oo", "ui": "oo-ee",
    "m": "m", "ng": "ng",
    "am": "ahm", "an": "ahn", "ang": "ahng",
    "em": "ehm", "en": "enn", "eng": "eng",
    "im": "eem", "in": "een", "ing": "ing",
    "om": "awm", "on": "awn", "ong": "awng",
    "um": "oom", "un": "oon", "ung": "oong",
}

def _strip_tone(syllable):
    return "".join(ch for ch in syllable if ch not in "123456")

def jyutping_to_approx(jp):
    """
    Very rough English-like approximation. Splits multi-syllable Jyutping by spaces,
    strips tone digits, and maps common finals to rough English hints.
    """
    if not jp:
        return ""
    parts = [p for p in jp.strip().split() if p]
    approx_parts = []
    for p in parts:
        base = _strip_tone(p)
        # try direct map first
        if base in _JP_APPROX_MAP:
            approx_parts.append(_JP_APPROX_MAP[base])
            continue
        # fallback: try to match common finals
        matched = False
        for k in sorted(_JP_APPROX_MAP.keys(), key=len, reverse=True):
            if base.endswith(k):
                approx_parts.append(_JP_APPROX_MAP[k])
                matched = True
                break
        if not matched:
            approx_parts.append(base)
    return " ".join(approx_parts)



#
# ------------------------------ macOS Voice Detection ------------------------------ #
_cached_say_voices = None

def _list_say_voices():
    """Return a list of lines from `say -v ?`, or [] if not available."""
    global _cached_say_voices
    if _cached_say_voices is not None:
        return _cached_say_voices
    try:
        out = subprocess.run(["say", "-v", "?"], check=False, capture_output=True, text=True)
        lines = out.stdout.splitlines() if out and out.stdout else []
        _cached_say_voices = lines
        return lines
    except Exception:
        _cached_say_voices = []
        return []

def _pick_cantonese_voice(preferred=None):
    """Pick a Cantonese-capable macOS voice. Try `preferred`, else match zh_HK/Cantonese, else None."""
    lines = _list_say_voices()
    if not lines:
        return None
    # If preferred provided and present, use it
    if preferred:
        for ln in lines:
            if ln.startswith(preferred + " ") or ln.split()[0] == preferred:
                return preferred
    # Try to find a voice that mentions Cantonese or zh_HK
    for ln in lines:
        low = ln.lower()
        if "cantonese" in low or "zh_hk" in low or "yue" in low:
            name = ln.split()[0]
            return name
    # Sometimes Sin-ji is available but not tagged; try common names
    for candidate in ("Sin-ji", "Tsz-Ho", "Sinji"):
        for ln in lines:
            if ln.startswith(candidate + " ") or ln.split()[0] == candidate:
                return candidate
    return None

# ------------------------------ Tooltips ------------------------------ #
class ToolTip(object):
    def __init__(self, widget, text=""):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        if self.tipwindow or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except Exception:
            return
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+{0}+{1}".format(x, y))
        label = tk.Label(tw, text=self.text, justify="left",
                         background="#FFFFE0", relief="solid", borderwidth=1,
                         font=("Helvetica", 10))
        label.pack(ipadx=6, ipady=3)

    def hide(self, event=None):
        tw = self.tipwindow
        if tw is not None:
            tw.destroy()
            self.tipwindow = None

TONE_DESCRIPTIONS = {
    "1": "High level",
    "2": "High rising",
    "3": "Mid level",
    "4": "Low falling",
    "5": "Low rising",
    "6": "Low level",
}

# --------------------------- Tone Colouring --------------------------- #

TONE_COLOURS = {
    "1": "#ADD8E6",  # High level: light sky blue
    "2": "#7FCDEB",  # High rising: more blue, less green
    "3": "#C8A2C8",  # Mid level: lavender
    "4": "#FFB347",  # Low falling: deep amber
    "5": "#FFE1B2",  # Low rising: lighter peach
    "6": "#E3C9A6",  # Low level: soft tan (slightly duller)
}

def tone_from_jyutping(jp):
    """
    Return the final tone digit for a Jyutping syllable string.
    If multiple syllables, take the first one's tone digit.
    """
    if not jp:
        return ""
    first = jp.strip().split()[0]
    for ch in reversed(first):
        if ch in "123456":
            return ch
    return ""

def colour_for_jyutping(jp):
    t = tone_from_jyutping(jp)
    return TONE_COLOURS.get(t, "")


# ------------------------------ TTS Helpers ------------------------------ #
VOICE_NAME_MAC = "Sin-ji"   # macOS Cantonese voice (zh_HK)
SAY_RATE_WPM = 180           # macOS 'say' words per minute
ESPEAK_VOICE = "zh-yue"     # eSpeak NG Cantonese voice id (fallback on non-macOS)

_def_voice_warned = False

def _tts_mac_say(text, voice, rate):
    global _def_voice_warned
    try:
        # Resolve a usable Cantonese voice
        chosen = _pick_cantonese_voice(preferred=voice or VOICE_NAME_MAC)
        r = str(rate if rate else SAY_RATE_WPM)
        if chosen:
            subprocess.run(["say", "-v", chosen, "-r", r, text], check=False)
        else:
            # Fallback: default system voice (may be English) so at least something is audible
            if not _def_voice_warned:
                print("[TTS] Cantonese voice not found; using default system voice.")
                _def_voice_warned = True
            subprocess.run(["say", "-r", r, text], check=False)
    except Exception as e:
        print("[TTS] macOS say error:", e)

def _tts_espeak(text, voice):
    try:
        v = voice or ESPEAK_VOICE
        try:
            subprocess.run(["espeak-ng", "-v", v, text], check=False)
        except FileNotFoundError:
            subprocess.run(["espeak", "-v", v, text], check=False)
    except Exception as e:
        print("[TTS] eSpeak error:", e)

def speak_text_async(text, voice=None, rate=None, enabled=True):
    """
    Speak the given Hanzi text in Cantonese without blocking the UI.
    On macOS uses 'say' with a Cantonese voice if available; otherwise tries eSpeak.
    """
    if not enabled or not text:
        return
    def _worker():
        try:
            if platform.system() == "Darwin":
                _tts_mac_say(text, voice, rate)
            else:
                _tts_espeak(text, voice)
        except Exception as e:
            print("[TTS] worker error:", e)
    threading.Thread(target=_worker, daemon=True).start()


def get_top_char_entries(top_n):
    """
    Count individual CJK characters; return top_n as list of dicts:
    [{"text": ch, "jyutping": jp}, ...]
    """
    tokens = load_hkcancor_tokens()
    big = "".join([t if isinstance(t, str) else str(t) for t in tokens])
    # Keep only CJK chars
    chars = [ch for ch in big if CJK_RE.match(ch)]
    counter = collections.Counter(chars)
    most_common = [ch for ch, _ in counter.most_common(top_n)]
    jp_list = pycantonese.characters_to_jyutping("".join(most_common))

    entries = []
    for ch, jp in zip(most_common, jp_list):
        jp_norm = _norm_jyut(jp)
        if jp_norm:
            entries.append({"text": ch, "jyutping": jp_norm})
    return entries


def get_top_word_entries(top_n, min_len=2, max_len=4):
    """
    Build CJK word tokens (contiguous CJK sequences), filter by length,
    and return top_n as list of dicts:
    [{"text": word, "jyutping": "char1_jp char2_jp ..."}, ...]
    """
    tokens = load_hkcancor_tokens()
    big = "".join([t if isinstance(t, str) else str(t) for t in tokens])
    # Extract contiguous CJK sequences (words-ish); then filter lengths
    seqs = CJK_RE.findall(big)
    words = []
    for s in seqs:
        # Split long runs into overlapping slices to approximate words
        # This is a heuristic; for better segmentation, use a real segmenter.
        if len(s) <= max_len:
            words.append(s)
        else:
            # chop into windows of length 2..max_len
            i = 0
            while i < len(s):
                # prefer max_len windows
                end = min(i + max_len, len(s))
                if end - i >= min_len:
                    words.append(s[i:end])
                i += 1

    # Keep only words of desired length
    words = [w for w in words if min_len <= len(w) <= max_len]

    counter = collections.Counter(words)
    most_common = [w for w, _ in counter.most_common(top_n)]

    # Map to Jyutping by char and join
    entries = []
    for w in most_common:
        jp_chars = pycantonese.characters_to_jyutping(w)
        if not jp_chars:
            continue
        jp_normed = [_norm_jyut(jp) for jp in jp_chars]
        jp_normed = [x for x in jp_normed if x]
        if not jp_normed or len(jp_normed) != len(w):
            continue
        jp_joined = " ".join(jp_normed)
        entries.append({"text": w, "jyutping": jp_joined})
    return entries


# ------------------------------- Tkinter UI ------------------------------ #

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title(APP_TITLE)

        # Dictionary load: direct paths (CC-Canto is guaranteed in assets/cc_canto.u8)
        self.cc_canto = load_cc_canto_dict(CC_CANTO_FILENAME)
        self.cedict = load_cedict_dict(DICT_FILENAME)
        if self.cc_canto and self.cedict:
            self.dict_name = "CC-Canto + CC-CEDICT ({0})".format(CC_CANTO_FILENAME)
        elif self.cc_canto:
            self.dict_name = "CC-Canto ({0})".format(CC_CANTO_FILENAME)
        elif self.cedict:
            self.dict_name = "CC-CEDICT"
        else:
            self.dict_name = "Mini-gloss only"

        # Controls
        ctrl = ttk.Frame(self, padding=10)
        ctrl.grid(row=0, column=0, sticky="ew")

        self.mode_var = tk.StringVar(value="characters")  # "characters" or "words"
        r1 = ttk.Radiobutton(ctrl, text="Characters", value="characters", variable=self.mode_var, command=self.rebuild_pool)
        r2 = ttk.Radiobutton(ctrl, text="Words", value="words", variable=self.mode_var, command=self.rebuild_pool)
        r1.grid(row=0, column=0, padx=(0,10))
        r2.grid(row=0, column=1, padx=(0,10))

        ttk.Label(ctrl, text="Top-N:").grid(row=0, column=2, padx=(10,4))
        self.topn_var = tk.IntVar(value=300)
        self.topn_spin = ttk.Spinbox(ctrl, from_=20, to=5000, increment=10, textvariable=self.topn_var, width=8, command=self.rebuild_pool)
        self.topn_spin.grid(row=0, column=3)

        self.shuffle_btn = ttk.Button(ctrl, text="Shuffle 5", command=self.shuffle)
        self.shuffle_btn.grid(row=0, column=4, padx=(10,0))

        # --- TTS Controls (Cantonese only, no voice dropdown) ---
        self.tts_enabled = tk.BooleanVar(value=True)
        self.rate_var = tk.IntVar()

        # Default rate
        self.rate_var.set(180)

        ttk.Checkbutton(ctrl, text="Speak on click", variable=self.tts_enabled).grid(row=0, column=5, padx=(12, 4))
        ttk.Label(ctrl, text="Rate").grid(row=0, column=6, padx=(8, 4))
        ttk.Spinbox(ctrl, from_=100, to=260, increment=10, textvariable=self.rate_var, width=5).grid(row=0, column=7)
        # Quick TTS test button (fixed Cantonese voice)
        self.tts_test_btn = ttk.Button(ctrl, text="Test Voice", command=lambda: speak_text_async(
            "廣東話你好", voice="Sin-ji", rate=self.rate_var.get(), enabled=self.tts_enabled.get()))
        self.tts_test_btn.grid(row=0, column=8, padx=(8,0))

        # Large Jyutping answer line (24pt) shown when a tile is clicked
        self.status_var = tk.StringVar()  # kept for compatibility, but no small label
        self.status_var.set("Jyutping: ")
        self.jp_answer = ttk.Label(ctrl, text="", font=("Helvetica", 24, "bold"))
        self.jp_answer.grid(row=1, column=0, columnspan=6, pady=(6, 0), sticky="w")

        # Tone legend bar (inside controls, under the big Jyutping line)
        legend = ttk.Frame(ctrl)
        legend.grid(row=2, column=0, columnspan=10, sticky="w", pady=(6, 0))
        ttk.Label(legend, text="Tone key:").grid(row=0, column=0, padx=(0, 8))
        for idx, tone in enumerate(["1", "2", "3", "4", "5", "6"], start=1):
            swatch = tk.Label(legend, text="{0}".format(tone), width=4, relief="solid", bd=1)
            swatch.configure(bg=TONE_COLOURS.get(tone, ""))
            swatch.grid(row=0, column=idx, padx=4, pady=2)
            # Tooltip with tone description
            try:
                ToolTip(swatch, TONE_DESCRIPTIONS.get(tone, ""))
            except Exception:
                pass

        # Grid for 1 × 5 tiles
        self.tile_frame = ttk.Frame(self, padding=10)
        self.tile_frame.grid(row=1, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.labels = []
        self.containers = []
        self.label_to_container = {}
        for col in range(5):
            # Outer container acts as a selectable border holder
            cont = tk.Frame(
                self.tile_frame,
                bd=0,
                highlightthickness=0,
                bg=self.cget("bg")
            )
            cont.grid(row=0, column=col, padx=8, pady=8, sticky="nsew")
            self.tile_frame.grid_columnconfigure(col, weight=1)

            lbl = tk.Label(
                cont,
                text="",
                font=("Helvetica", 44),
                width=3,
                padx=24,
                pady=18,
                borderwidth=1,
                relief="solid"
            )
            # Pack inside the container with small padding so the container colour shows as a border
            lbl.pack(fill="both", expand=True, padx=2, pady=2)

            self.labels.append(lbl)
            self.containers.append(cont)
            self.label_to_container[lbl] = cont

        # Details box below the grid with a thin border and title "DETAILS"
        details_frame = tk.LabelFrame(self, text="DETAILS", bd=1, relief="solid", labelanchor="nw", padx=6, pady=6)
        details_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0,10))
        self.details = scrolledtext.ScrolledText(details_frame, height=6, wrap="word")
        self.details.configure(font=("Helvetica", 16))
        self.details.pack(fill="both", expand=True)
        self.grid_rowconfigure(2, weight=1)

        self.pool = []
        self.current_five = []
        self.selected_label = None
        self.rebuild_pool()
        self.shuffle()

    def _clear_selection(self):
        if self.selected_label is not None:
            try:
                cont = self.label_to_container.get(self.selected_label)
                if cont is not None:
                    cont.configure(bg=self.cget("bg"))
            except Exception:
                pass
            self.selected_label = None

    def _select_label(self, lbl):
        # Clear previous selection and highlight the new one
        self._clear_selection()
        try:
            cont = self.label_to_container.get(lbl)
            if cont is not None:
                cont.configure(bg="#000000")  # black border background
        except Exception:
            pass
        self.selected_label = lbl

    def rebuild_pool(self):
        """
        Build the candidate pool based on UI (mode + top_n).
        """
        mode = self.mode_var.get()
        topn = self.topn_var.get()
        try:
            if mode == "characters":
                self.pool = get_top_char_entries(topn)
            else:
                self.pool = get_top_word_entries(topn, min_len=2, max_len=4)
        except Exception as e:
            messagebox.showerror("Error", "Failed to build pool:\n{0}".format(e))
            self.pool = []
            return

    def shuffle(self):
        # Clear any previous selection highlight
        self._clear_selection()
        if hasattr(self, "jp_answer"):
            self.jp_answer.configure(text="")
        if not self.pool:
            self.rebuild_pool()
        if len(self.pool) < 5:
            messagebox.showwarning("Not enough items", "Pool has fewer than 5 items.")
            return
        self.current_five = random.sample(self.pool, 5)
        for idx, e in enumerate(self.current_five):
            lbl = self.labels[idx]
            lbl.configure(text=e["text"])
            # Reset unselected visuals
            cont = self.containers[idx]
            try:
                cont.configure(bg=self.cget("bg"))
            except Exception:
                pass
            lbl.configure(bd=1, relief="solid")
            bg = colour_for_jyutping(e["jyutping"])
            if bg:
                lbl.configure(bg=bg)
            else:
                lbl.configure(bg=self.cget("bg"))
            # Tooltip showing tone explanation for this tile
            try:
                tone_digit = tone_from_jyutping(e["jyutping"]) or ""
                tip_text = TONE_DESCRIPTIONS.get(tone_digit, "")
                if tip_text:
                    ToolTip(lbl, tip_text)
            except Exception:
                pass
            lbl.bind("<Button-1>", self._make_click_handler(e))

    def _make_click_handler(self, entry):
        def handler(event):
            text = entry["text"]
            jp = _norm_jyut(entry["jyutping"])
            meanings = lookup_meaning_merged(text, self.cc_canto, self.cedict)
            meaning_str = "; ".join(meanings[:6])  # keep popup concise
            self.status_var.set("Jyutping: {0}".format(jp))
            try:
                self.jp_answer.configure(text=jp)
            except Exception:
                pass
            # Speak the clicked Hanzi/word in Cantonese (non-blocking) using UI settings
            try:
                speak_text_async(
                    text,
                    voice="Sin-ji",
                    rate=self.rate_var.get(),
                    enabled=self.tts_enabled.get()
                )
            except Exception:
                pass
            # Tone-based recolour (kept)
            bg = colour_for_jyutping(jp)
            if bg:
                event.widget.configure(bg=bg)
            # Highlight the clicked tile (no dialog)
            self._select_label(event.widget)
            # Clear and append details into the text box (without the Jyutping line)
            approx = jyutping_to_approx(jp)
            self.details.delete("1.0", tk.END)
            self.details.insert(tk.END, "English approximation: {0}\n".format(approx))
            self.details.insert(tk.END, "Meaning(s): {0}\n".format(meaning_str))
        return handler


def main():
    try:
        app = App()
        app.minsize(820, 380)
        app.mainloop()
    except Exception as e:
        # If HKCanCor needs to download and fails, show a helpful message
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Startup Error", "Problem starting the app:\n{0}\n\nTip: The first call to pycantonese.hkcancor() may download the corpus.".format(e))

if __name__ == "__main__":
    main()
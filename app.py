# -*- coding: utf-8 -*-
import collections
import os
# Audio helpers for TTS (non-blocking)
import platform
import random
import re
import subprocess
import threading
import tkinter as tk
import warnings
from tkinter import ttk, messagebox, scrolledtext

from constants import BOTH_CHAR_RATIO
from messages import PLAY_MODE_MESSAGES

# --- Import TTS settings from settings.py, with safe defaults if missing ---
try:
    from settings import SPEAK_ON_CLICK, TTS_RATE
except Exception:
    SPEAK_ON_CLICK = True  # default: speak when tile is clicked
    TTS_RATE = 180  # default macOS say rate (wpm)

# Silence noisy UserWarnings emitted by wordseg/pkg_resources during import
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*"
)

# Optional: simplified conversion using OpenCC (free). If not installed, we silently skip.
try:
    from opencc import OpenCC

    _opencc_t2s = OpenCC('t2s')
except Exception:
    _opencc_t2s = None


def to_simplified(text: str) -> str:
    """Convert Traditional → Simplified if OpenCC is available; otherwise return input."""
    try:
        if _opencc_t2s:
            return _opencc_t2s.convert(text)
    except Exception:
        pass
    return text


import pycantonese

from dictionaries import MINI_GLOSS, ANDYS_LIST, TRICKY_INITIALS

# CAPP_TITLE = "Cantonese (HKCanCor) – 1×5 with Meanings"
APP_TITLE = "Cantonese (HKCanCor) – 1×5 with Meanings"
CJK_RE = re.compile(u"[\u4E00-\u9FFF]+")
DICT_FILENAME = os.path.join("assets", "cedict_ts.u8")  # CC-CEDICT in assets/
CC_CANTO_FILENAME = os.path.join("assets", "cc_canto.u8")  # CC-Canto in assets/

# Divider label for mode dropdown
DIVIDER_LABEL = "──────────"


# ------------------------ Dictionary (CC-family) ------------------------ #

def _load_cedict_like(path):
    """
    Load a CEDICT-like file into {traditional: [glosses...]}. Lines look like:
      傳統 簡体 [pin1 yin1] /meaning 1/meaning 2/...
    Returns {} if file not present or unreadable.
    """
    if not os.path.exists(path):
        return {}
    out = {}
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
                    bucket = out.get(trad, [])
                    for m in meanings:
                        if m not in bucket:
                            bucket.append(m)
                    out[trad] = bucket
    except Exception:
        pass
    return out


def load_cedict_dict(path):
    return _load_cedict_like(path)


def load_cc_canto_dict(path):
    return _load_cedict_like(path)


# ------------------ Merged dictionary lookup (CC-Canto > CEDICT > MINI_GLOSS > char comp) ------------------ #


def _greedy_seg(word: str, dict_keys: set[str]) -> list[str]:
    """Greedy left-to-right longest-match segmentation using the provided dictionary keys."""
    i = 0
    parts: list[str] = []
    n = len(word)
    while i < n:
        matched = None
        # Try the longest substring starting at i
        for j in range(n, i, -1):
            cand = word[i:j]
            if cand in dict_keys:
                matched = cand
                break
        if matched is None:
            matched = word[i:i + 1]  # fallback: single character
        parts.append(matched)
        i += len(matched)
    return parts


def lookup_meaning_merged(hanzi, cc_canto, cedict):
    defs = []

    # 1) CC-Canto first
    defs += cc_canto.get(hanzi, [])

    # 2) Then CEDICT (append, not replace)
    defs += cedict.get(hanzi, [])

    # 3) Then MINI_GLOSS (append)
    if hanzi in MINI_GLOSS:
        defs += MINI_GLOSS[hanzi]

    # 4) Then ANDYS_LIST (append)
    if hanzi in ANDYS_LIST:
        defs += ANDYS_LIST[hanzi]

    # De-duplicate while preserving order
    seen = set()
    merged = []
    for d in defs:
        d_norm = d.strip()
        if d_norm and d_norm not in seen:
            seen.add(d_norm)
            merged.append(d_norm)

    if merged:
        return merged

    # --- Free fallbacks for multi-character words ---
    if isinstance(hanzi, str) and len(hanzi) > 1:
        # 1) Try simplified form (if OpenCC available)
        simp = to_simplified(hanzi)
        if simp and simp != hanzi:
            simp_defs = []
            simp_defs += cc_canto.get(simp, [])
            simp_defs += cedict.get(simp, [])
            for s in simp_defs:
                s = s.strip()
                if s and s not in seen:
                    seen.add(s)
                    merged.append(s)
            if merged:
                return merged

        # 2) Greedy longest-match segmentation with existing dictionary keys (traditional)
        dict_keys = set(cc_canto.keys()) | set(cedict.keys()) | set(MINI_GLOSS.keys())
        segs = _greedy_seg(hanzi, dict_keys)
        seg_glosses = []
        for seg in segs:
            g = []
            g += cc_canto.get(seg, [])
            g += cedict.get(seg, [])
            if seg in MINI_GLOSS:
                g += MINI_GLOSS[seg]
            if not g and len(seg) == 1 and seg in ANDYS_LIST:
                g += ANDYS_LIST[seg]
            if g:
                seg_glosses.append(seg + ": " + "; ".join(g[:2]))
            else:
                seg_glosses.append(seg + ": (no entry)")
        if seg_glosses:
            return [" + ".join(seg_glosses)]

    return ["(meaning not available)"]


# ------------------ Grammar label extraction & hints ------------------ #
_LABEL_RE = re.compile(r"^\(([^)]+)\)\s*(.*)$")

# Extra hints for single characters commonly used as particles/markers in Cantonese
GRAMMAR_HINTS = {
    "的": ["structural particle", "possessive marker"],
    "嘅": ["particle", "possessive/structural (Cantonese)"],
    "了": ["aspect particle", "completed action"],
    "咗": ["aspect particle (Cantonese)", "perfective"],
    "過": ["experiential aspect (after verbs)"],
    "著": ["durative/continuous aspect (literary)", "stative marker"],
    "緊": ["progressive aspect (Cantonese)"],
    "嗎": ["question particle"],
    "呢": ["continuative/question particle"],
    "吧": ["suggestion/softener particle"],
    "呀": ["sentence-final particle (Cantonese)", "exclamatory"],
    "啦": ["sentence-final particle (Cantonese)", "imperative/urging"],
    "喇": ["sentence-final particle (Cantonese)", "change-of-state"],
    "囉": ["sentence-final particle (Cantonese)", "assertive"],
    "個": ["classifier (general)"],
    "啲": ["classifier (plural/small amount; Cantonese)"],
    "條": ["classifier (long, thin)"],
    "隻": ["classifier (animals; one of a pair)"],
    "件": ["classifier (clothes, matters)"],
    "唔": ["adverb", "negation (Cantonese)"],
    "不": ["adverb", "negation (literary/Mandarin)"],
    "冇": ["verb", "existential negation (Cantonese)"]
}


def extract_labels_and_clean(glosses, hanzi):
    """Extract leading parenthetical labels from each gloss and return (labels, cleaned_glosses).
    Also add GRAMMAR_HINTS for single characters when applicable.
    """
    labels = []
    cleaned = []
    for g in glosses:
        m = _LABEL_RE.match(g)
        if m:
            label, rest = m.groups()
            # split multiple labels if separated by ',' or ';'
            for tok in re.split(r",|;", label):
                tok = tok.strip()
                if tok and tok not in labels:
                    labels.append(tok)
            g = rest.strip()
        cleaned.append(g)
    # Single-character extra hints
    if isinstance(hanzi, str) and len(hanzi) == 1:
        hints = GRAMMAR_HINTS.get(hanzi)
        if hints:
            for h in hints:
                if h not in labels:
                    labels.append(h)
    return labels, cleaned


# --- Heuristic POS inference from English glosses ---
def _infer_pos_labels(glosses):
    """Heuristically infer POS labels from English glosses.
    Allows mixed classes (e.g., noun + verb) when evidence appears in different senses.
    """
    labels = set()
    noun_candidate = False  # mark if any sense looks nominal without explicit POS keywords
    for g in glosses:
        if not g:
            continue
        gl = g.strip().lower()
        matched_specific = False
        # Strong keyword matches first
        if "classifier" in gl or "measure word" in gl:
            labels.add("classifier")
            matched_specific = True
        # Particle subtypes
        if "question particle" in gl:
            labels.add("question particle")
            matched_specific = True
        if "sentence-final" in gl or "final particle" in gl:
            labels.add("sentence-final particle")
            matched_specific = True
        if "structural" in gl and "particle" in gl:
            labels.add("structural particle")
            matched_specific = True
        if "particle" in gl and "aspect" in gl:
            labels.add("aspect particle")
            matched_specific = True
        elif "particle" in gl:
            labels.add("particle")
            matched_specific = True
        if "interjection" in gl or "exclamation" in gl:
            labels.add("interjection")
            matched_specific = True
        if "pronoun" in gl:
            labels.add("pronoun")
            matched_specific = True
        if "preposition" in gl:
            labels.add("preposition")
            matched_specific = True
        if "conjunction" in gl:
            labels.add("conjunction")
            matched_specific = True
        if "adverb" in gl or gl.startswith("adv. "):
            labels.add("adverb")
            matched_specific = True
        # Verb heuristic: many glosses start verbs with "to ..."
        if gl.startswith("to ") or gl.startswith("v. "):
            labels.add("verb")
            matched_specific = True
        # Adjective markers
        if "adjective" in gl or gl.startswith("adj. "):
            labels.add("adjective")
            matched_specific = True
        if "proper noun" in gl:
            labels.add("proper noun")
            matched_specific = True
        if "surname" in gl:
            labels.add("surname")
            matched_specific = True
        if "numeral" in gl:
            labels.add("numeral")
            matched_specific = True
        # If this sense didn't match any explicit POS keyword and doesn't look like a verb, treat as a noun candidate
        if not matched_specific and not gl.startswith("to "):
            noun_candidate = True

    # Add noun if any sense looked nominal, or if nothing matched at all
    if noun_candidate or not labels:
        labels.add("noun")
    return list(labels)


_POS_ORDER = [
    "noun", "proper noun", "surname", "numeral",
    "verb", "adjective", "adverb",
    "pronoun", "preposition", "conjunction", "interjection",
    "classifier",
    "structural particle", "question particle", "sentence-final particle", "aspect particle", "particle",
]


def _sort_labels(labels):
    order = {k: i for i, k in enumerate(_POS_ORDER)}
    return sorted(labels, key=lambda x: order.get(x, 999))


def format_character_meanings(hanzi, glosses):
    """Return a single formatted string like '字: (labels) 1. sense; 2. sense' for single characters."""
    labels, cleaned = extract_labels_and_clean(glosses, hanzi)
    # If no explicit labels were found, infer from the cleaned glosses
    if not labels:
        inferred = _infer_pos_labels(cleaned)
        for lab in inferred:
            if lab not in labels:
                labels.append(lab)
    numbered = "; ".join(f"{i}. {g}" for i, g in enumerate(cleaned[:6], 1) if g)
    if labels:
        return f"{hanzi}: (" + "; ".join(labels) + ") " + numbered
    else:
        return f"{hanzi}: " + numbered


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


def _sentence_text_from_tokens(tokens):
    """Join a list of tokens into a readable sentence string."""
    try:
        return "".join(tokens)
    except Exception:
        return " ".join(str(t) for t in tokens)


def _jyutping_for_text(text: str) -> str:
    """Return a rough Jyutping line by mapping per character and joining with spaces."""
    try:
        jps = pycantonese.characters_to_jyutping(text) or []
        out = []
        for jp in jps:
            norm = _norm_jyut(jp)
            if norm:
                out.append(norm)
        return " ".join(out)
    except Exception:
        return ""


def find_hkcancor_examples(term: str, max_examples: int = 2):
    """Find up to max_examples sentences from HKCanCor containing the term.
    Returns list of (text, jyutping_line).
    """
    examples = []
    try:
        corpus = pycantonese.hkcancor()
        # Prefer sents() if available
        try:
            for sent in corpus.sents():
                try:
                    txt = _sentence_text_from_tokens(sent)
                except Exception:
                    txt = "".join(sent) if isinstance(sent, (list, tuple)) else str(sent)
                if term and txt and term in txt:
                    jp_line = _jyutping_for_text(txt)
                    examples.append((txt, jp_line))
                    if len(examples) >= max_examples:
                        break
        except Exception:
            # Fallback to utterances()
            for utt in corpus.utterances():
                txt = getattr(utt, "transcript_text", "")
                if term and txt and term in txt:
                    jp_line = _jyutping_for_text(txt)
                    examples.append((txt, jp_line))
                    if len(examples) >= max_examples:
                        break
    except Exception:
        pass
    return examples


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

# ---------------------- Improved English Approximation ---------------------- #

_INITIAL_MAP = {
    "b": "b", "p": "p", "m": "m", "f": "f",
    "d": "d", "t": "t", "n": "n", "l": "l",
    "g": "g", "k": "k", "ng": "ng", "h": "h",
    "z": "z",  # Jyutping z ≈ unaspirated ts (dz/ts); use "z" for clarity
    "c": "ts",  # Jyutping c ≈ aspirated ts; simplified to "ts"
    "s": "s",
    "gw": "gw", "kw": "kw",
    "w": "w",
    "j": "y",  # Jyutping j ~ English 'y'
}

_RIME_MAP = {
    "aa": "ah", "aai": "eye", "aau": "ow",
    "a": "ah", "ai": "eye", "au": "ow",
    "am": "ahm", "an": "ahn", "ang": "ahng",
    "ap": "ahp", "at": "aht", "ak": "ahk",

    "e": "eh", "ei": "ay", "em": "em", "en": "en", "eng": "eng",
    "ep": "ep", "et": "et", "ek": "ek",
    "eoi": "oey", "eo": "er", "oe": "ur",

    "i": "ee", "iu": "yoo",
    "im": "eem", "in": "een", "ing": "ing",
    "ip": "eep", "it": "eet", "ik": "eek",

    "o": "aw", "oi": "oy", "ou": "oh",
    "om": "awm", "on": "awn", "ong": "awng",
    "op": "awp", "ot": "awt", "ok": "awk",

    "u": "oo", "ui": "oo-ee",
    "um": "oom", "un": "oon", "ung": "oong",
    "up": "oop", "ut": "oot", "uk": "ook",

    "m": "m", "ng": "ng",
}

# ordered list of initials for longest match
_INITIALS = sorted(_INITIAL_MAP.keys(), key=len, reverse=True)


def _strip_tone(syllable: str) -> str:
    return "".join(ch for ch in syllable if ch not in "123456")


def _split_initial_rime(base: str):
    for ini in _INITIALS:
        if base.startswith(ini):
            return ini, base[len(ini):]
    return "", base


def jyutping_to_approx(jp: str) -> str:
    """Convert Jyutping to a more complete English-like hint."""
    if not jp:
        return ""
    syllables = [p for p in jp.strip().split() if p]
    outputs = []
    for syl in syllables:
        base = _strip_tone(syl).lower()
        ini, rime = _split_initial_rime(base)
        ini_hint = _INITIAL_MAP.get(ini, ini)
        rime_hint = _RIME_MAP.get(rime, rime)
        outputs.append((ini_hint + rime_hint).strip())
    return "-".join(outputs)


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
        # Add these two lines:
        self.widget.bind("<FocusIn>", self.show)
        self.widget.bind("<FocusOut>", self.hide)

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
        # Add this so it shows above other windows on macOS:
        try:
            tw.wm_attributes("-topmost", True)
        except Exception:
            pass
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
    "1": "#9CC5D6",  # High level: slightly duller sky blue
    "2": "#8AD6F0",  # High rising: marginally brighter blue
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
VOICE_NAME_MAC = "Sin-ji"  # macOS Cantonese voice (zh_HK)
SAY_RATE_WPM = 180  # macOS 'say' words per minute
ESPEAK_VOICE = "zh-yue"  # eSpeak NG Cantonese voice id (fallback on non-macOS)

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


def _big_cjk_string_from_tokens(tokens):
    return "".join([t if isinstance(t, str) else str(t) for t in tokens])


def get_top_char_entries(top_n):
    """
    Count individual CJK characters; return top_n as list of dicts:
    [{"text": ch, "jyutping": jp}, ...]
    """
    tokens = load_hkcancor_tokens()
    big = _big_cjk_string_from_tokens(tokens)
    # Keep only CJK chars
    chars = [ch for ch in big if CJK_RE.match(ch)]
    counter = collections.Counter(chars)
    most_common = [ch for ch, _ in counter.most_common(top_n)]

    entries = []
    for ch in most_common:
        try:
            jps = pycantonese.characters_to_jyutping(ch)
            jp_norm = _norm_jyut(jps[0] if jps else "")
            if jp_norm:
                entries.append({"text": ch, "jyutping": jp_norm})
        except Exception:
            continue
    return entries


def get_top_word_entries(top_n, min_len=2, max_len=4):
    """
    Build CJK word tokens (contiguous CJK sequences), filter by length,
    and return top_n as list of dicts:
    [{"text": word, "jyutping": "char1_jp char2_jp ..."}, ...]
    """
    tokens = load_hkcancor_tokens()
    big = _big_cjk_string_from_tokens(tokens)
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


# Helper for MINI_GLOSS entries for "Very common" mode
def get_minigloss_entries():
    """Return entries from MINI_GLOSS: [{"text": key, "jyutping": joined_jp}]"""
    return entries_from_gloss_dict(MINI_GLOSS)


# Generic builder for MINI_GLOSS-style dicts
def entries_from_gloss_dict(gloss_dict):
    """Return entries from a MINI_GLOSS-style dict: [{"text": hanzi, "jyutping": joined_jp}]"""
    entries = []
    for key in gloss_dict.keys():
        jp_chars = pycantonese.characters_to_jyutping(key)
        jp_normed = [_norm_jyut(jp) for jp in jp_chars] if jp_chars else []
        jp_normed = [x for x in jp_normed if x]
        jp_joined = " ".join(jp_normed) if jp_normed else ""
        entries.append({"text": key, "jyutping": jp_joined})
    return entries


# ------------------------------- Tkinter UI ------------------------------ #

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title(APP_TITLE)

        # Initialize runtime state early to avoid AttributeError before first shuffle
        self.selected_label = None
        self.pool = []
        self.current_five = []

        # Audio-first mode gating
        self.require_audio_before_selection = False
        self.has_played_for_round = True  # standard mode allows selection immediately
        # Track whether the play-mode hint is currently visible
        self.mode_msg_visible = False

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

        self.mode_var = tk.StringVar(value="Andy's List")
        self._last_mode_label = "Minimal Common"
        self.mode_combo = ttk.Combobox(
            ctrl,
            values=["Andy's List", "Minimal Common", "Tricky Initials", DIVIDER_LABEL, "Characters", "Words", "Both"],
            textvariable=self.mode_var,
            state="readonly",
            width=14,
        )
        self.mode_combo.grid(row=0, column=0, padx=(0, 10))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_combo_selected)

        self.top_label = ttk.Label(ctrl, text="Words:")
        self.top_label.grid(row=0, column=2, padx=(10, 4))
        DEFAULT_TOPN = 300
        self.topn_var = tk.IntVar(value=DEFAULT_TOPN)
        self.topn_spin = ttk.Spinbox(ctrl, from_=20, to=5000, increment=10, textvariable=self.topn_var, width=5,
                                     command=self.rebuild_pool)
        self.topn_spin.grid(row=0, column=3)
        # Initialize for Minimal Common: show total and disable spin
        self.topn_var.set(len(ANDYS_LIST))
        self.topn_spin.configure(state="disabled")

        self.shuffle_btn = ttk.Button(ctrl, text="Shuffle", command=self.shuffle)
        self.shuffle_btn.grid(row=0, column=4, padx=(10, 0))

        # Play mode: Listen & Choose (default) vs Hear Pronunciation
        self.play_mode_var = tk.StringVar(value="Listen & Choose")
        self.play_mode_combo = ttk.Combobox(
            ctrl,
            values=["Listen & Choose", "Hear Pronunciation"],
            state="readonly",
            width=24,
            textvariable=self.play_mode_var,
        )
        # Move Play sound button before mode combobox (column 5)
        # --- Audio Controls (no TTS UI; use settings.py) ---
        # Play button container (no focus ring/highlight management)
        self.play_container = tk.Frame(
            ctrl,
            bd=0,
            relief="flat",
            padx=0,
            pady=0,
        )
        self.play_container.grid(row=0, column=5, padx=(8, 0))

        self.make_sound_btn = ttk.Button(self.play_container, text="Play sound", width=10, takefocus=1, command=self._on_make_sound)
        self.make_sound_btn.pack(fill="both", expand=True)

        self.play_mode_combo.grid(row=0, column=6, padx=(8, 0))
        self.play_mode_combo.bind("<<ComboboxSelected>>", self._on_play_mode_change)

        # Quick TTS test button (fixed Cantonese voice)
        self.tts_test_btn = ttk.Button(ctrl, text="Test Voice", command=lambda: speak_text_async(
            "廣東話你好", voice="Sin-ji", rate=TTS_RATE, enabled=SPEAK_ON_CLICK))
        self.tts_test_btn.grid(row=0, column=7, padx=(8, 0))

        # --- Static UI under the control row ---
        # Large Jyutping answer line (24pt)
        self.status_var = tk.StringVar(value="Jyutping: ")
        self.jp_answer = ttk.Label(ctrl, text="", font=("Helvetica", 24, "bold"))
        self.jp_answer.grid(row=1, column=0, columnspan=5, pady=(6, 0), sticky="w")
        # Play-mode message aligned with the Play sound button (now at column 5)
        # Instructions textbox aligned with the Play sound button (column 5)
        self.instructions_box = tk.Text(
            ctrl,
            height=2,
            width=70,
            font=("Helvetica", 16),
            wrap="word",
            state="disabled",
            relief="flat",
            bg=self.cget("bg")
        )
        self.instructions_box.grid(row=1, column=5, columnspan=3, padx=(0, 0), sticky="w")

        # Tone legend bar (inside controls, under the big Jyutping line)
        self.legend_frame = ttk.Frame(ctrl)
        self.legend_frame.grid(row=2, column=0, columnspan=10, sticky="w", pady=(6, 0))
        ttk.Label(self.legend_frame, text="Tone key:").grid(row=0, column=0, padx=(0, 8))
        for idx, tone in enumerate(["1", "2", "3", "4", "5", "6"], start=1):
            swatch = tk.Label(self.legend_frame, text=f"{tone}", width=4, relief="solid", bd=1)
            swatch.configure(bg=TONE_COLOURS.get(tone, ""))
            swatch.grid(row=0, column=idx, padx=4, pady=2)
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
        self.label_tips = {}
        self.overlays = {}
        for col in range(5):
            cont = tk.Frame(
                self.tile_frame,
                bd=0,
                highlightthickness=0,
                bg=self.cget("bg")
            )
            cont.grid(row=0, column=col, padx=8, pady=8, sticky="nsew")
            self.tile_frame.grid_columnconfigure(col, weight=1, minsize=200)

            lbl = tk.Label(
                cont,
                text="",
                font=("Helvetica", 44),
                width=6,
                padx=24,
                pady=18,
                borderwidth=1,
                relief="solid"
            )
            lbl.pack(fill="both", expand=True, padx=2, pady=2)

            tip = ToolTip(lbl, "")
            self.label_tips[lbl] = tip

            # Small overlay in the top-right corner for ✓ / ✗ feedback
            ov = tk.Label(
                cont,
                text="",
                font=("Helvetica", 22, "bold"),
                fg="#2e7d32",  # green (correct) / will switch to red for wrong
                bg=lbl.cget("bg"),
                borderwidth=0,
                padx=0,
                pady=0,
            )
            # Place slightly inset at the top-right of the tile container
            ov.place(relx=1.0, rely=0.0, anchor="ne", x=-4, y=4)
            self.overlays[lbl] = ov

            self.labels.append(lbl)
            self.containers.append(cont)
            self.label_to_container[lbl] = cont

        # Details box below the grid with a thin border and title "DETAILS"
        details_frame = tk.LabelFrame(self, text="DETAILS", bd=1, relief="solid", labelanchor="nw", padx=6, pady=6)
        details_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.details = scrolledtext.ScrolledText(details_frame, height=9, wrap="word")
        self.details.configure(font=("Helvetica", 16))
        self.details.pack(fill="both", expand=True)
        self._current_initial_for_help = ""
        self.grid_rowconfigure(2, weight=1)

        # Track the target text for Recognise pronunciation correctness check
        self.target_text = None

        # Build initial pool and populate tiles on startup
        try:
            self.rebuild_pool()
            self.shuffle()
        except Exception:
            pass

        # Apply initial mode state (Standard disables Play a sound)
        self._on_play_mode_change()
        # Show appropriate mode message on first run
        self._show_instructions_message()
        # If default mode is Listen & Choose, focus Play after UI is realized (ring appears via FocusIn)
        try:
            if (self.play_mode_var.get() or "").strip().lower() == "listen & choose":
                self.after(80, lambda: self.make_sound_btn.focus_set())
        except Exception:
            pass

    def _on_play_mode_change(self, event=None):
        """Switch between Hear Pronunciation and Listen & Choose modes (enable/disable Play button and set gating)."""
        play_mode = (self.play_mode_var.get() or "hear pronunciation").strip().lower()
        if play_mode == "listen & choose":
            try:
                self.make_sound_btn.configure(state="normal")
            except Exception:
                pass
            self.require_audio_before_selection = True
            self.has_played_for_round = False
        else:
            try:
                self.make_sound_btn.configure(state="disabled")
            except Exception:
                pass
            self.require_audio_before_selection = False
            self.has_played_for_round = True
        # Unhighlight any current tile selection when switching modes
        try:
            self._clear_selection()
        except Exception:
            pass
        # Clear blue selection highlight in the Combobox entry (macOS)
        try:
            self.play_mode_combo.selection_clear()
            self.play_mode_combo.icursor('end')
        except Exception:
            pass

        # Shuffle tiles automatically on mode change
        try:
            self.shuffle()
        except Exception:
            pass

        # Move focus depending on mode (no focus ring management)
        try:
            if play_mode == "listen & choose":
                self.make_sound_btn.focus_set()
            else:
                self.shuffle_btn.focus_set()
        except Exception:
            pass

        # Update play-mode help text using central messages and mark visible
        self._show_instructions_message()

    def _show_instructions_message(self, text: str | None = None):
        """Show an instructions message in the instructions_box; if text is None, pick from current mode."""
        try:
            if text is None:
                mode = (self.play_mode_var.get() or "").strip().lower()
                if mode == "listen & choose":
                    text = PLAY_MODE_MESSAGES["play_mode"][0]
                else:
                    text = PLAY_MODE_MESSAGES["play_mode"][1]
            self.instructions_box.configure(state="normal")
            self.instructions_box.delete("1.0", tk.END)
            self.instructions_box.insert(tk.END, text)
            self.instructions_box.configure(state="disabled")
            self.mode_msg_visible = True
        except Exception:
            pass

    def _hide_instructions_message(self):
        """Clear the instructions textbox without removing the widget."""
        try:
            self.instructions_box.configure(state="normal")
            self.instructions_box.delete("1.0", tk.END)
            self.instructions_box.configure(state="disabled")
            self.mode_msg_visible = False
        except Exception:
            pass




    def _derive_initial_from_jp(self, jp: str) -> str:
        try:
            base = "".join(ch for ch in jp if ch not in "123456").lower()
            ini, _ = _split_initial_rime(base)
            return ini
        except Exception:
            return ""

    # _on_sound_explain removed; explanation now shown directly in details

    def _clear_selection(self):
        if getattr(self, "selected_label", None) is not None:
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

    def _current_list_mode(self):
        v = self.mode_var.get()
        return {
            "Minimal Common": "very_common",
            "Andy's List": "andys",
            "Tricky Initials": "tricky",
            "Characters": "characters",
            "Words": "words",
            "Both": "both",
        }.get(v, "very_common")

    def _on_combo_selected(self, event=None):
        label = self.mode_var.get()
        if label == DIVIDER_LABEL:
            # Revert to the last valid selection without triggering a mode change
            self.mode_var.set(self._last_mode_label)
            return
        # Update last valid label and proceed
        self._last_mode_label = label
        self._on_mode_change()

    def _on_mode_change(self):
        """Handle mode dropdown changes: update labels/spinbox, rebuild pool, and shuffle."""
        mode = self._current_list_mode()
        if mode in ("very_common", "andys", "tricky"):
            # Show total count and disable the Top spinbox for fixed dictionaries
            self.top_label.configure(text="Words:")
            if mode == "very_common":
                total = len(MINI_GLOSS)
            elif mode == "andys":
                total = len(ANDYS_LIST)
            else:  # tricky
                total = len(TRICKY_INITIALS)
            self.topn_var.set(total)
            self.topn_spin.configure(state="disabled")
        else:
            # Character/Word/Both modes: enable Top spinbox
            self.top_label.configure(text="Top:")
            if self.topn_var.get() in (len(MINI_GLOSS), len(ANDYS_LIST), len(TRICKY_INITIALS)):
                self.topn_var.set(300)
            self.topn_spin.configure(state="normal")
        # Rebuild and refresh tiles
        self.rebuild_pool()
        self.shuffle()

    def _on_make_sound(self):
        """Play the pronunciation of one randomly chosen tile from the current five."""
        try:
            # Ensure we have a set of tiles to choose from
            if not self.current_five:
                self.shuffle()
            choices = [e for e in (self.current_five or []) if e and e.get("text")]
            if not choices:
                messagebox.showinfo("Info", "No tiles available. Try Shuffle first.")
                return
            chosen = random.choice(choices)
            text = chosen.get("text", "")
            if not text:
                messagebox.showinfo("Info", "No valid selection to play.")
                return
            # Remember the target text for Listen & Choose correctness check
            self.target_text = text
            speak_text_async(
                text,
                voice="Sin-ji",
                rate=TTS_RATE,
                enabled=SPEAK_ON_CLICK,
            )
            # Allow selection for this round in Listen & Choose mode
            self.has_played_for_round = True
            # --- Custom logic: print NUMBER_OF_CHANCES if mode is Listen & Choose ---
            try:
                mode = (self.play_mode_var.get() or "").strip().lower()
                if mode == "listen & choose":
                    try:
                        from settings import NUMBER_OF_CHANCES
                        print("NUMBER_OF_CHANCES:", NUMBER_OF_CHANCES)
                    except Exception:
                        pass
                if mode == "listen & choose" and self.mode_msg_visible:
                    self._hide_instructions_message()
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Could not play sound: {e}")

    def rebuild_pool(self):
        """
        Build the candidate pool based on UI (mode + top_n).
        """
        mode = self._current_list_mode()
        topn = self.topn_var.get()
        try:
            if mode == "very_common":
                self.pool = get_minigloss_entries()
            elif mode == "andys":
                self.pool = entries_from_gloss_dict(ANDYS_LIST)
            elif mode == "tricky":
                self.pool = entries_from_gloss_dict(TRICKY_INITIALS)
            elif mode == "characters":
                self.pool = get_top_char_entries(topn)
            elif mode == "both":
                # Split Top between characters and words based on BOTH_CHAR_RATIO
                n_chars = int(topn * BOTH_CHAR_RATIO)
                n_words = topn - n_chars
                pool_chars = get_top_char_entries(n_chars)
                pool_words = get_top_word_entries(n_words, min_len=2, max_len=4)
                self.pool = pool_chars + pool_words
            else:
                self.pool = get_top_word_entries(topn, min_len=2, max_len=4)
        except Exception as e:
            messagebox.showerror("Error", "Failed to build pool:\n{0}".format(e))
            self.pool = []
            return

    def shuffle(self):
        # Clear any previous selection highlight
        self._clear_selection()
        # Clear all overlay marks
        try:
            for _lbl, _ov in self.overlays.items():
                _ov.configure(text="")
        except Exception:
            pass
        # Reset target
        self.target_text = None
        if hasattr(self, "jp_answer"):
            self.jp_answer.configure(text="")
            # Clear the Details box
        if hasattr(self, "details"):
            self.details.delete("1.0", tk.END)
        # Reset audio-first gate each shuffle
        if self.require_audio_before_selection:
            self.has_played_for_round = False
        else:
            self.has_played_for_round = True
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
            # Sync overlay background and clear any previous mark
            try:
                ov = self.overlays.get(lbl)
                if ov is not None:
                    ov.configure(text="")
                    ov.configure(bg=lbl.cget("bg"))
            except Exception:
                pass
            # Tooltip showing tone explanation for this tile (update existing)
            try:
                tone_digit = tone_from_jyutping(e["jyutping"]) or ""
                tip_text = TONE_DESCRIPTIONS.get(tone_digit, "")
                tip = self.label_tips.get(lbl)
                if tip is not None:
                    tip.text = tip_text
                else:
                    # Fallback if missing for any reason
                    self.label_tips[lbl] = ToolTip(lbl, tip_text)
            except Exception:
                pass
            lbl.bind("<Button-1>", self._make_click_handler(e))

    def _render_meanings_block(self, text, meanings, is_single_char, add_service_note, examples):
        """Write meanings (and labels if single char), then examples, then note, into DETAILS."""
        if is_single_char:
            labels, cleaned = extract_labels_and_clean(meanings, text)
            if not labels:
                labels = _infer_pos_labels(cleaned)
            labels = _sort_labels(labels)
            if labels:
                self.details.insert(tk.END, "Labels: {0}\n".format("; ".join(labels)))
            self.details.insert(tk.END, "Meaning(s):\n")
            for i, g in enumerate(cleaned[:6], 1):
                if g:
                    self.details.insert(tk.END, "  {0}. {1}\n".format(i, g))
        else:
            self.details.insert(tk.END, "Meaning(s):\n")
            for i, g in enumerate(meanings[:6], 1):
                if g:
                    self.details.insert(tk.END, "  {0}. {1}\n".format(i, g))
        # Examples before note
        if add_service_note and examples:
            self.details.insert(tk.END, "Usage (HKCanCor):\n")
            for (sent_txt, sent_jp) in examples:
                self.details.insert(tk.END, f"  • {sent_txt}\n")
                if sent_jp:
                    self.details.insert(tk.END, f"    {sent_jp}\n")
        if add_service_note:
            self.details.insert(tk.END, "  An Azure or Google service account is required to translate this.\n")

    def _make_click_handler(self, entry):
        def handler(event):
            # In Listen & Choose mode, require the user to play a random sound before selection
            if self.require_audio_before_selection and not self.has_played_for_round:
                messagebox.showinfo("Recognise pronunciation", "Click ‘Play’ before selecting a tile.")
                return
            # In Hear Pronunciation mode, hide the hint as soon as a tile is selected
            try:
                mode = (self.play_mode_var.get() or "").strip().lower()
                if mode == "hear pronunciation" and self.mode_msg_visible:
                    self._hide_instructions_message()
            except Exception:
                pass
            text = entry["text"]
            jp = _norm_jyut(entry["jyutping"])  # initial value from pool
            # For single characters, recompute directly to avoid any misalignment
            if isinstance(text, str) and len(text) == 1:
                try:
                    jps = pycantonese.characters_to_jyutping(text)
                    if jps:
                        jp_click = _norm_jyut(jps[0])
                        if jp_click:
                            jp = jp_click
                except Exception:
                    pass

            # Clear previous overlay marks
            # (Persistence: do NOT clear overlays here, so previous ticks/crosses remain visible)
            # try:
            #     for _lbl, _ov in self.overlays.items():
            #         _ov.configure(text="")
            # except Exception:
            #     pass

            # In Listen & Choose mode, show tick/cross depending on correctness (only after Play)
            try:
                play_mode = (self.play_mode_var.get() or "").strip().lower()
                if play_mode == "listen & choose" and self.has_played_for_round:
                    ov = self.overlays.get(event.widget)
                    if ov is not None:
                        if getattr(self, "target_text", None) and text == self.target_text:
                            ov.configure(text="✓", fg="#2e7d32")  # green tick
                        else:
                            ov.configure(text="✗", fg="#c62828")  # red cross
                        ov.lift()
            except Exception:
                pass

            # Insert sound explanation directly for tricky initials
            ini = self._derive_initial_from_jp(jp)
            self._current_initial_for_help = ini

            meanings = lookup_meaning_merged(text, self.cc_canto, self.cedict)

            # Update big Jyutping answer
            try:
                self.jp_answer.configure(text=jp)
            except Exception:
                pass

            # Speak the clicked Hanzi/word in Cantonese (non-blocking) using settings.py config
            try:
                speak_text_async(
                    text,
                    voice="Sin-ji",
                    rate=TTS_RATE,
                    enabled=SPEAK_ON_CLICK,
                )
            except Exception:
                pass

            # Tone-based recolour (kept)
            bg = colour_for_jyutping(jp)
            if bg:
                event.widget.configure(bg=bg)

            # Highlight the clicked tile (no dialog)
            self._select_label(event.widget)

            # Clear and append details into the text box as a vertical list
            approx = jyutping_to_approx(jp)
            self.details.delete("1.0", tk.END)

            # One-line header: English approximation only
            self.details.insert(
                tk.END,
                "English approximation: {0}\n".format(approx)
            )
            # If meanings is exactly "(meaning not available)", add a note and collect examples
            add_service_note = False
            examples = []  # initialize to avoid any scope issues
            if meanings == ["(meaning not available)"]:
                add_service_note = True
                # Try usage examples from HKCanCor
                try:
                    examples = find_hkcancor_examples(text, max_examples=2)
                except Exception:
                    examples = []

            is_single_char = isinstance(text, str) and len(text) == 1
            self._render_meanings_block(text, meanings, is_single_char, add_service_note, examples)

            # Directly append explanation for tricky initials after meanings block
            if ini in {"z", "c", "j", "ng"}:
                self.details.insert(tk.END, "---\n")
                explanations = {
                    "z": (
                        "Jyutping z",
                        "In Jyutping, z represents an unaspirated alveolar affricate /ts/ (between English 'dz' in 'kids' and 'ts' in 'cats').\n\n"
                        "Many learners prefer to write this as 'z', 'dz', or 'ts' for clarity."
                    ),
                    "c": (
                        "Jyutping c",
                        "In Jyutping, c represents an aspirated alveolar affricate /tsʰ/ — like 'ts' with a puff of air (as in 'cats', but stronger).\n\n"
                        "It's often approximated as 'ts' (or 'ts(h)')."
                    ),
                    "j": (
                        "Jyutping j",
                        "In Jyutping, j is the glide /j/, like English 'y' in 'yes' — not English 'j' (/dʒ/).\n\n"
                        "So 'ji' is close to 'yee'."
                    ),
                    "ng": (
                        "Jyutping ng",
                        "A syllable-initial /ŋ/ (the 'ng' in 'sing'), but at the start of a syllable. English rarely begins words with this, so it can feel unusual."
                    ),
                }
                title, msg = explanations[ini]
                self.details.insert(tk.END, f"{title}:\n{msg}\n")

        return handler


def main():
    try:
        app = App()
        app.minsize(1100, 480)
        app.mainloop()
    except Exception as e:
        # If HKCanCor needs to download and fails, show a helpful message
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Startup Error",
                             "Problem starting the app:\n{0}\n\nTip: The first call to pycantonese.hkcancor() may download the corpus.".format(
                                 e))


if __name__ == "__main__":
    main()

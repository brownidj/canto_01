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

from constants import (BOTH_CHAR_RATIO,
                       TONE_DESCRIPTIONS,
                       TONE_COLOURS,
                       TONE_KEY_TEXT_COLOURS,
                       TOOLTIP_DELAY,
                       PLAY_TOOLTIP,
                       PLAYMODE_TOOLTIP,
                       MODE_TOOLTIP,
                       TOPN_TOOLTIP,)

LIGHT_TILE_BG = "#F5F5F5"  # very light grey, tone-neutral
from messages import PLAY_MODE_MESSAGES, RESULT_MESSAGES, SHUFFLE_MESSAGE, DUPLICATE_WARNING

# --- Import TTS settings from settings.py, with safe defaults if missing ---
try:
    from settings import SPEAK_ON_CLICK, TTS_RATE
except Exception:
    SPEAK_ON_CLICK = True  # default: speak when tile is clicked
    TTS_RATE = 180  # default macOS say rate (wpm)

 # --- Import NUMBER_OF_CHANCES from settings.py, with a safe default ---
try:
    from settings import NUMBER_OF_CHANCES
except Exception:
    NUMBER_OF_CHANCES = 3

# --- Import tricky initials explanations flag from settings.py, with a safe default ---
try:
    from settings import TRICKY_INITIAL_EXPLANATIONS
except Exception:
    TRICKY_INITIAL_EXPLANATIONS = False


# --- Import DEBUG from settings.py, with a safe default ---
try:
    from settings import DEBUG
except Exception:
    DEBUG = False


# --- Import default play mode from settings.py, with a safe default ---
try:
    from settings import PLAY_MODE_DEFAULT
except Exception:
    PLAY_MODE_DEFAULT = "Pronunciation"

# --- Import default Jyutping mode from settings.py, with a safe fallback ---
try:
    from settings import JYUTPING_MODE_DEFAULT
except Exception:
    JYUTPING_MODE_DEFAULT = "Strict"


# --- Import Jyutping display formatting options from settings.py, with safe defaults ---
try:
    from settings import JYUTPING_STYLE, JYUTPING_WORD_BOUNDARY_MARKER
except Exception:
    JYUTPING_STYLE = "learner"  # "learner" | "strict" | "strict_with_word_boundaries"
    JYUTPING_WORD_BOUNDARY_MARKER = " · "

try:
    from config_manager import load_config, save_config
except Exception:
    # Provide safe fallbacks: load returns a dict; save accepts one argument
    load_config = lambda: {}
    save_config = lambda cfg: None
    print("WARNING: config_manager.py not found; using in-memory defaults only.")
# --- Import default play mode from settings.py, with a safe default ---
try:
    from settings import PLAY_MODE_DEFAULT
except Exception:
    PLAY_MODE_DEFAULT = "Pronunciation"

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
# --- Robust Jyutping syllable splitter ---
_JP_SYL_RE = re.compile(r"[a-z]+[1-6]", flags=re.IGNORECASE)

def _safe_split_syllables(jp_chunk: str) -> list[str]:
    """
    Split a Jyutping chunk into syllables. Handles:
    - Space-separated forms: "gwong2 bo3"
    - Concatenated forms:   "gwong2bo3"
    Returns a list like ["gwong2", "bo3"].
    """
    if not jp_chunk:
        return []
    jp_chunk = _norm_jyut(jp_chunk).strip()
    if not jp_chunk:
        return []
    # Fast path: spaces already present
    parts = [p for p in jp_chunk.split() if p]
    if len(parts) >= 2:
        return parts
    # Regex fallback: find all letter+tone groups
    found = _JP_SYL_RE.findall(jp_chunk)
    if found:
        return found
    return parts if parts else [jp_chunk]


def to_simplified(text: str) -> str:
    """Convert Traditional → Simplified if OpenCC is available; otherwise return input."""
    try:
        if _opencc_t2s:
            return _opencc_t2s.convert(text)
    except Exception:
        pass
    return text


import pycantonese

from dictionaries import MINI_GLOSS, ANDYS_LIST, TRICKY_INITIALS, TEST_WORDS

APP_TITLE = "Cantonese (HKCanCor) – Five tiles with Meanings"
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

# --- Jyutping display formatting helper ---
def _format_jyutping_for_display(text: str, fallback_jp: str) -> str:
    """
    Format Jyutping according to settings:
    - learner: syllables spaced; words separated by single space.
    - strict: join syllables within each word; words separated by single space.
    - strict_with_word_boundaries: join syllables within each word; words separated by a visible marker.
    Uses pycantonese to obtain per-word Jyutping when possible; falls back to the provided jp string.
    """
    style = (JYUTPING_STYLE or "learner").strip().lower()
    marker = JYUTPING_WORD_BOUNDARY_MARKER if JYUTPING_WORD_BOUNDARY_MARKER is not None else " · "

    # 1) Try to segment into lexicon words first, so "strict" can join within each segment
    segments = None
    try:
        if hasattr(pycantonese, "word_segment"):
            segments = pycantonese.word_segment(text)  # list of strings
        elif hasattr(pycantonese, "segment"):
            segments = pycantonese.segment(text)  # fallback API name if present
    except Exception:
        segments = None

    words = []
    try:
        if isinstance(segments, (list, tuple)) and len(segments) >= 2:
            # Build per-segment jyutping strings (syllables spaced inside each segment)
            tmp = []
            for seg in segments:
                try:
                    seg_jps = pycantonese.characters_to_jyutping(seg) or []
                except Exception:
                    seg_jps = []
                # normalize and keep syllables spaced inside this segment
                norm = [_norm_jyut(j) for j in seg_jps if _norm_jyut(j)]
                if norm:
                    tmp.append(" ".join(norm))
                else:
                    # If no jyutping for a segment, try per-char as last resort
                    try:
                        per_char = pycantonese.characters_to_jyutping("".join(seg)) or []
                        norm2 = [_norm_jyut(j) for j in per_char if _norm_jyut(j)]
                        tmp.append(" ".join(norm2))
                    except Exception:
                        pass
            words = [w for w in tmp if w]
        if not words:
            # 2) Fall back to pycantonese's own word-level jyutping for the whole text
            try:
                words = pycantonese.characters_to_jyutping(text) or []
            except Exception:
                words = []
            words = [_norm_jyut(w) for w in words if _norm_jyut(w)]
        if not words:
            # 3) Final fallback: use the provided jp string as a single "word"
            words = [_norm_jyut(fallback_jp)] if fallback_jp else []
    except Exception:
        # extremely defensive: ensure words is a list of strings
        try:
            words = [_norm_jyut(fallback_jp)] if fallback_jp else []
        except Exception:
            words = []

    # Debug trace (optional)
    try:
        if DEBUG:
            seg_count = (len(segments) if isinstance(segments, (list, tuple)) else "n/a")
            print(f"[DBG] jpfmt: seg={seg_count} words={len(words)} style={style}")
    except Exception:
        pass

    # Style application
    if style == "strict":
        # Remove spaces inside each word; keep a single space between segmented words
        return " ".join(w.replace(" ", "") for w in words if w)
    if style == "strict_with_word_boundaries":
        return str(marker).join(w.replace(" ", "") for w in words if w)
    # default learner: keep syllable spaces and a single space between words
    return " ".join(words)


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
        self._after_id = None
        self.widget.bind("<Enter>", self._schedule_show)
        self.widget.bind("<Leave>", self._cancel_scheduled_show)
        self.widget.bind("<FocusIn>", self._schedule_show)
        self.widget.bind("<FocusOut>", self._cancel_scheduled_show)

    # NEW: works for both ttk and classic Tk widgets
    def _is_disabled(self) -> bool:
        try:
            # ttk widgets: prefer instate()
            if hasattr(self.widget, "instate"):
                return self.widget.instate(("disabled",))
        except Exception:
            pass
        try:
            # classic Tk widgets
            return str(self.widget.cget("state")).lower() == "disabled"
        except Exception:
            return False

    def _schedule_show(self, event=None):
        # 🚫 Do not schedule if disabled
        if self._is_disabled():
            return
        self._cancel_scheduled_show()
        self._after_id = self.widget.after(TOOLTIP_DELAY, lambda: self.show(event))

    def _cancel_scheduled_show(self, event=None):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self.hide()

    def show(self, event=None):
        if not self.text:
            return
        # Extra safety: don't show if disabled (covers race conditions)
        if self._is_disabled():
            return
        # If the widget isn't viewable yet (e.g., right after startup), try again shortly
        try:
            if not self.widget.winfo_viewable():
                self.widget.after(150, lambda: self.show(event))
                return
        except Exception:
            pass
        # If already showing, destroy and recreate for consistent behavior
        if self.tipwindow:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
            self.tipwindow = None
        # Compute position; prefer event root coords if available, else widget geometry, else pointer
        try:
            if event is not None and hasattr(event, 'x_root') and hasattr(event, 'y_root'):
                x = event.x_root + 10
                y = event.y_root + 10
            else:
                x = self.widget.winfo_rootx() + 10
                y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except Exception:
            try:
                x = self.widget.winfo_pointerx() + 10
                y = self.widget.winfo_pointery() + 10
            except Exception:
                return
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
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
        # Listen & Choose chances tracking
        self.remaining_chances = NUMBER_OF_CHANCES
        self.first_selection_message_shown = False
        # Track whether the play-mode hint is currently visible
        self.mode_msg_visible = False
        # Track wrong attempts in current Listen & Choose round
        self._wrong_this_round = set()

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
        # Ensure the controls row (row 0) is tall enough to fit three radio buttons
        try:
            ctrl.grid_rowconfigure(0, minsize=48)  # adjust 90–130 if you change fonts
        except Exception:
            pass
        # Make Jyutping answer (left) expand, Tone key (right) fixed
        ctrl.grid_columnconfigure(0, weight=0)
        ctrl.grid_columnconfigure(1, weight=1)
        # try:
        #     # Ensure highlighted columns have a measurable width for debug borders
        #     ctrl.grid_columnconfigure(2, minsize=0)
        #     ctrl.grid_columnconfigure(3, minsize=0)
        # except Exception:
        #     pass

        self.word_list_var = tk.StringVar(value="Andy's List")
        self._last_mode_label = "Minimal Common"

        # --- Containerized layout inside ctrl ---
        basic_ctrls_container = ttk.Frame(ctrl)
        basic_ctrls_container.grid(row=0, column=0, columnspan=2, rowspan=1, sticky="nw")  # Columns 1–2, Row 1
        # Do not let col 0 expand; keep both cols natural so controls stay together on the left
        basic_ctrls_container.grid_columnconfigure(0, weight=0)
        basic_ctrls_container.grid_columnconfigure(1, weight=0)
        # Allow for multiple rows: controls row and instructions row inside basic_ctrls_container
        basic_ctrls_container.grid_rowconfigure(0, weight=0)
        basic_ctrls_container.grid_rowconfigure(1, weight=0)

        play_mode_container = ttk.Frame(ctrl)
        play_mode_container.grid(row=0, column=2, rowspan=1, sticky="nw", padx=(8, 0))  # Column 3, Row 1 only

        settings_container = ttk.Frame(ctrl)
        settings_container.grid(row=0, column=3, columnspan=2, rowspan=1, sticky="ne", padx=(10, 0))  # Columns 4–5, Row 1
        settings_container.grid_columnconfigure(0, weight=1)

        # --- Combo + Words/Top + Spinbox: all inside a frame for clean alignment ---
        combo_frame = tk.Frame(basic_ctrls_container)
        combo_frame.grid(row=0, column=0, sticky="nw", padx=(0, 0))

        self.word_list_combo = ttk.Combobox(
            combo_frame,
            values=["Andy's List", "Minimal Common", "Tricky Initials", "Test Words", DIVIDER_LABEL, "Characters", "Words", "Both"],
            textvariable=self.word_list_var,
            state="readonly",
            width=14,
        )
        self.word_list_combo.pack(side="left")
        ToolTip(self.word_list_combo, MODE_TOOLTIP)
        self.word_list_combo.bind("<<ComboboxSelected>>", self._on_combo_selected)

        self.words_or_top_label = ttk.Label(combo_frame, text="Words:")
        self.words_or_top_label.pack(side="left", padx=(10, 4))
        DEFAULT_TOPN = 300
        self.top_words_number = tk.IntVar(value=DEFAULT_TOPN)
        self.top_spin_number = ttk.Spinbox(combo_frame, from_=20, to=5000, increment=10, textvariable=self.top_words_number, width=5,
                                           command=self.rebuild_pool)
        self.top_spin_number.pack(side="left")
        ToolTip(self.top_spin_number, TOPN_TOOLTIP)
        # Initialize for Minimal Common: show total and disable spin
        self.top_words_number.set(len(ANDYS_LIST))
        self.top_spin_number.configure(state="disabled")

        # --- Column 2: Shuffle and Play sound (left-aligned together) ---
        col2_frame = tk.Frame(basic_ctrls_container)
        col2_frame.grid(row=0, column=1, sticky="nw", padx=(10, 0))

        # --- Instructions container: now inside basic_ctrls_container, row=1 ---
        instructions_container = tk.Frame(basic_ctrls_container, highlightthickness=0, bd=0, takefocus=0)
        instructions_container.grid(row=1, column=0, columnspan=2, sticky="nw", pady=(6, 0))
        instructions_container.grid_columnconfigure(0, weight=1)
        try:
            instructions_container.grid_propagate(True)
        except Exception:
            pass

        self.shuffle_btn = ttk.Button(col2_frame, text="Shuffle", command=self.shuffle)
        self.shuffle_btn.pack(side="left")
        ToolTip(self.shuffle_btn, SHUFFLE_MESSAGE)

        # --- Audio Controls (no TTS UI; use settings.py) ---
        # Play button container (no focus ring/highlight management)
        self.play_container = tk.Frame(
            col2_frame,
            bd=0,
            relief="flat",
            padx=0,
            pady=0,
        )
        self.play_container.pack(side="left", padx=(8, 0))

        self.make_sound_btn = ttk.Button(self.play_container, text="Play sound", width=10, takefocus=1, command=self._on_make_sound)
        self.make_sound_btn.pack(fill="both", expand=True)
        ToolTip(self.make_sound_btn, PLAY_TOOLTIP)

        # --- Play mode radios: group inside play_mode_container ---
        # Determine initial play mode from config (if any) or fallback to settings default
        try:
            cfg = load_config()
        except Exception:
            cfg = {}
        try:
            last_mode = cfg.get("CURRENT_PLAY_MODE") if isinstance(cfg, dict) else None
        except Exception:
            last_mode = None
        initial_mode = last_mode or PLAY_MODE_DEFAULT
        self.play_mode_var = tk.StringVar(value=initial_mode)

        # Reflect the initial mode back into settings so other modules can read it
        try:
            import settings as _settings_mod
            _settings_mod.CURRENT_PLAY_MODE = self.play_mode_var.get()
        except Exception:
            pass

        # Create a LabelFrame for play mode radios
        radios_frame = tk.LabelFrame(play_mode_container, text="Mode", bd=1, relief="solid", labelanchor="nw", padx=6, pady=6)
        radios_frame.grid(row=0, column=0, sticky="nw")

        rb1 = ttk.Radiobutton(
            radios_frame,
            text="Pronunciation",
            variable=self.play_mode_var,
            value="Pronunciation",
            command=self._on_play_mode_change,
        )
        rb2 = ttk.Radiobutton(
            radios_frame,
            text="Listen & Choose",
            variable=self.play_mode_var,
            value="Listen & Choose",
            command=self._on_play_mode_change,
        )
        rb3 = ttk.Radiobutton(
            radios_frame,
            text="Future Option",
            variable=self.play_mode_var,
            value="Future Option",
            command=self._on_play_mode_change,
        )

        rb1.configure(takefocus=1)
        rb2.configure(takefocus=1)
        rb3.configure(takefocus=1)

        rb1.pack(anchor="w")
        rb2.pack(anchor="w")
        rb3.pack(anchor="w")

        ToolTip(rb1, PLAYMODE_TOOLTIP)
        ToolTip(rb2, PLAYMODE_TOOLTIP)
        ToolTip(rb3, PLAYMODE_TOOLTIP)

        # --- Column 4: Rate spinbox (100–260 in steps of 10) ---
        col4_frame = tk.Frame(settings_container)
        col4_frame.grid(row=0, column=1, sticky="e", padx=(10, 0))

        self.rate_label = ttk.Label(col4_frame, text="TTS Rate:")
        self.rate_label.pack(side="left", padx=(0, 4))

        try:
            default_rate = int(TTS_RATE)
        except Exception:
            default_rate = 180

        self.rate_var = tk.IntVar(value=default_rate)
        self.rate_spin = ttk.Spinbox(
            col4_frame,
            from_=100,
            to=260,
            increment=10,
            textvariable=self.rate_var,
            width=5
        )
        self.rate_spin.pack(side="left")

        # --- Jyutping display mode radios (Learner / Strict / Borders) ---
        jyut_frame = tk.LabelFrame(settings_container, text="Jyutping", bd=1, relief="solid", labelanchor="nw", padx=6, pady=6)
        jyut_frame.grid(row=0, column=2, sticky="e", padx=(10, 0))

        # Map settings style → UI label and back
        _style_to_label = {
            "learner": "Learner",
            "strict": "Strict",
            "strict_with_word_boundaries": "Borders",
        }
        _label_to_style = {v: k for k, v in _style_to_label.items()}

        # Load last-saved Jyutping mode from config, or fall back to settings default
        try:
            _cfg = load_config()
        except Exception:
            _cfg = {}
        try:
            _last_jp_mode = _cfg.get("CURRENT_JYUTPING_MODE") if isinstance(_cfg, dict) else None
        except Exception:
            _last_jp_mode = None
        _initial_jp_mode = _last_jp_mode or JYUTPING_MODE_DEFAULT  # UI label: "Learner" | "Strict" | "Borders"

        # Set the UI variable to the label directly
        self.jyutping_style_var = tk.StringVar(value=_initial_jp_mode)

        # Also sync the underlying style string used by rendering
        try:
            globals()["JYUTPING_STYLE"] = _label_to_style.get(_initial_jp_mode, "learner")
        except Exception:
            pass
        # Reflect initial mode into settings so other modules can read it
        try:
            import settings as _settings_mod
            _settings_mod.CURRENT_JYUTPING_MODE = _initial_jp_mode
        except Exception:
            pass

        def _on_jp_style_change():
            # Update the global style so rendering picks it up
            try:
                new_style = _label_to_style.get(self.jyutping_style_var.get(), "learner")
                globals()["JYUTPING_STYLE"] = new_style
            except Exception:
                pass
            # Persist the chosen Jyutping mode label to settings and config.json
            try:
                import settings as _settings_mod
                _settings_mod.CURRENT_JYUTPING_MODE = self.jyutping_style_var.get()
            except Exception:
                pass
            try:
                _cfg = load_config()
                if not isinstance(_cfg, dict):
                    _cfg = {}
                _cfg["CURRENT_JYUTPING_MODE"] = self.jyutping_style_var.get()
                save_config(_cfg)
            except Exception:
                pass
            # Refresh the Jyutping answer display if present
            try:
                # If there is a current selection, re-render that; otherwise, render the first tile if available
                text_to_render = None
                if getattr(self, "selected_label", None) is not None:
                    try:
                        text_to_render = self.selected_label.cget("text")
                    except Exception:
                        text_to_render = None
                if not text_to_render and self.labels:
                    try:
                        text_to_render = self.labels[0].cget("text")
                    except Exception:
                        text_to_render = None
                if text_to_render:
                    self._render_jyutping_colored(text_to_render, "")
            except Exception:
                pass

        # Three radio buttons
        rb_jp_learner = ttk.Radiobutton(jyut_frame, text="Learner", variable=self.jyutping_style_var, value="Learner", command=_on_jp_style_change)
        rb_jp_strict = ttk.Radiobutton(jyut_frame, text="Strict", variable=self.jyutping_style_var, value="Strict", command=_on_jp_style_change)
        rb_jp_borders = ttk.Radiobutton(jyut_frame, text="Borders", variable=self.jyutping_style_var, value="Borders", command=_on_jp_style_change)

        rb_jp_learner.pack(anchor="w")
        rb_jp_strict.pack(anchor="w")
        rb_jp_borders.pack(anchor="w")

        # --- Instructions & messages container inside ctrl (Columns 1–2, Row 2) ---
        instr_container = ttk.Frame(instructions_container)
        instr_container.grid(row=0, column=0, sticky="nw", padx=(0, 0), pady=(0, 0))
        try:
            instr_container.grid_columnconfigure(0, weight=1)
            instr_container.grid_propagate(True)
        except Exception:
            pass

        self.instructions_box = tk.Text(
            instr_container,
            height=2,
            width=56,
            font=("Helvetica", 24),
            wrap="word",
            state="disabled",
            relief="flat",
            bg=self.cget("bg"),
            takefocus=0,  # <- add this
            highlightthickness=0  # <- ensures no border if clicked
        )
        self.instructions_box.grid(row=0, column=0, sticky="nw")
        try:
            self.instructions_box.tag_configure("left", justify="left")
        except Exception:
            pass

        # --- Debug overlay for ctrl grid ---

        # --- Status row below ctrl: tone legend (left) + Jyutping answer (right) ---
        status = ttk.Frame(self)
        status.grid(row=1, column=0, sticky="ew")
        status.grid_columnconfigure(0, weight=0)
        status.grid_columnconfigure(1, weight=1)

        self.legend_frame = tk.Frame(status)
        self.legend_frame.grid(row=0, column=0, sticky="w", pady=(6, 0))

        self.status_var = tk.StringVar(value="Jyutping: ")
        self.jp_answer_frame = tk.Frame(status, bg=self.cget("bg"))
        self.jp_answer_frame.grid(row=0, column=1, sticky="sew", padx=(30, 0))
        self.jp_answer_frame.pack_propagate(False)
        self.jp_answer_frame.configure(height=60)

        # Recreate swatches in legend_frame
        for idx, tone in enumerate(["1", "2", "3", "4", "5", "6"]):
            fg = TONE_KEY_TEXT_COLOURS.get(str(tone), "#000000")
            bg = TONE_COLOURS.get(str(tone))
            swatch = tk.Label(self.legend_frame, text=f"{tone}", width=2, padx=10, relief="solid", bd=1, bg=bg, fg=fg)
            swatch.grid(row=0, column=idx, padx=4, pady=(21, 15))
            try:
                ToolTip(swatch, TONE_DESCRIPTIONS.get(tone, ""))
            except Exception:
                pass

        try:
            self.legend_frame.update_idletasks()
            needed_w = self.legend_frame.winfo_reqwidth()
            self.legend_frame.configure(width=needed_w, height=60)
            self.legend_frame.grid_propagate(False)
        except Exception:
            pass

        # Grid for 1 × 5 tiles
        self.tile_frame = ttk.Frame(self, padding=10)
        self.tile_frame.grid(row=2, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.labels = []
        self.containers = []
        self.label_to_container = {}
        self.label_tips = {}
        self.overlays = {}
        for col in range(5):
            cont = tk.Frame(
                self.tile_frame,
                bd=0,
                highlightthickness=2,            # reserve space from the start
                highlightbackground=self.cget("bg"),  # same as background → invisible initially
                bg=self.cget("bg"),
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

            try:
                lbl.configure(bg=LIGHT_TILE_BG)
            except Exception:
                pass

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
        details_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.details = scrolledtext.ScrolledText(details_frame, height=9, wrap=tk.WORD)
        self.details.configure(font=("Helvetica", 16))
        self.details.pack(fill="both", expand=True)
        self._current_initial_for_help = ""
        self.grid_rowconfigure(3, weight=1)
        # Ensure ctrl has 5 columns, with column 1 expandable
        try:
            for _c in range(5):
                ctrl.grid_columnconfigure(_c, weight=1 if _c == 1 else 0)
        except Exception:
            pass

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
        # Apply initial focus based on current mode (shows blue outline on Play if Listen & Choose)
        self._apply_mode_focus()

    def _dbg(self, *args):
        try:
            if DEBUG:
                print("[DBG]", *args)
        except Exception:
            pass

    def _apply_mode_focus(self):
        """Give focus to Play (Listen & Choose) or Shuffle (Pronunciation) after idle so the focus ring is shown."""
        try:
            mode = (self.play_mode_var.get() or "").strip().lower()
            if mode == "listen & choose":
                # Schedule after idle so the widget is realized and the blue outline is drawn
                self.after_idle(lambda: self.make_sound_btn.focus_set())
            else:
                self.after_idle(lambda: self.shuffle_btn.focus_set())
        except Exception:
            pass

    def _on_play_mode_change(self, event=None):
        """Switch between Pronunciation and Listen & Choose modes (enable/disable Play button and set gating)."""
        play_mode = (self.play_mode_var.get() or "Pronunciation").strip().lower()
        # Persist the current play mode to settings and config.json
        current_mode = self.play_mode_var.get()
        try:
            import settings as _settings_mod
            _settings_mod.CURRENT_PLAY_MODE = current_mode
        except Exception:
            pass

        try:
            cfg = load_config()
            cfg["CURRENT_PLAY_MODE"] = current_mode
            save_config(cfg)
        except Exception:
            pass
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

        # Shuffle tiles automatically on mode change
        try:
            self.shuffle()
        except Exception:
            pass

        # Move focus depending on mode (ensures blue outline appears on Play in Listen & Choose)
        self._apply_mode_focus()

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
            self._dbg("_show_instructions_message:", repr(text))
            self.instructions_box.configure(state="normal")
            self.instructions_box.delete("1.0", tk.END)
            self.instructions_box.insert(tk.END, (text or "") + "\n", "left")
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
                    cont.configure(bg=self.cget("bg"), highlightbackground=self.cget("bg"))  # keep thickness, hide border
            except Exception:
                pass
            self.selected_label = None

    def _select_label(self, lbl):
        # Clear previous selection and highlight the new one
        self._clear_selection()
        try:
            cont = self.label_to_container.get(lbl)
            if cont is not None:
                # Border space already reserved; just change colour to reveal it
                cont.configure(highlightbackground="#000000", bg=self.cget("bg"))
        except Exception:
            pass
        self.selected_label = lbl

    def _current_list_mode(self):
        v = self.word_list_var.get()
        return {
            "Minimal Common": "very_common",
            "Andy's List": "andys",
            "Tricky Initials": "tricky",
            "Test Words": "test_words",
            "Characters": "characters",
            "Words": "words",
            "Both": "both",
        }.get(v, "very_common")

    def _on_combo_selected(self, event=None):
        label = self.word_list_var.get()
        if label == DIVIDER_LABEL:
            # Revert to the last valid selection without triggering a mode change
            self.word_list_var.set(self._last_mode_label)
            return
        # Update last valid label and proceed
        self._last_mode_label = label
        self._on_mode_change()

    def _on_mode_change(self):
        """Handle mode dropdown changes: update labels/spinbox, rebuild pool, and shuffle."""
        mode = self._current_list_mode()
        if mode in ("very_common", "andys", "tricky", "test_words"):
            # Show total count and disable the Top spinbox for fixed dictionaries
            self.words_or_top_label.configure(text="Words:")
            if mode == "very_common":
                total = len(MINI_GLOSS)
            elif mode == "andys":
                total = len(ANDYS_LIST)
            elif mode == "tricky":
                total = len(TRICKY_INITIALS)
            else:  # test_words
                total = len(TEST_WORDS)
            self.top_words_number.set(total)
            self.top_spin_number.configure(state="disabled")
        else:
            # Character/Word/Both modes: enable Top spinbox
            self.words_or_top_label.configure(text="Top:")
            if self.top_words_number.get() in (len(MINI_GLOSS), len(ANDYS_LIST), len(TRICKY_INITIALS)):
                self.top_words_number.set(300)
            self.top_spin_number.configure(state="normal")
        # Rebuild and refresh tiles
        self.rebuild_pool()
        self.shuffle()

    def _on_make_sound(self):
        """Play the pronunciation for the current round.
        - In Listen & Choose: replay the existing target if the round is active; otherwise start a new round and pick a new target.
        - In Pronunciation: button is disabled anyway.
        """
        try:
            # Ensure we have tiles to choose from
            if not self.current_five:
                self.shuffle()

            choices = [e for e in (self.current_five or []) if e and e.get("text")]
            if not choices:
                messagebox.showinfo("Info", "No tiles available. Try Shuffle first.")
                return

            mode = (self.play_mode_var.get() or "").strip().lower()

            # Default: no text chosen yet
            text_to_play = None

            if mode == "listen & choose":
                # If a round is already active and we have a target, just replay it
                if self.has_played_for_round and self.target_text:
                    text_to_play = self.target_text
                    self._dbg("replay target:", self.target_text)
                else:
                    # Start a NEW round: clear overlays, pick a NEW target
                    self._clear_overlays()
                    chosen = random.choice(choices)
                    text_to_play = chosen.get("text", "")
                    self._dbg("new round target:", text_to_play)
                    if not text_to_play:
                        messagebox.showinfo("Info", "No valid selection to play.")
                        return

                    # Remember the new target and reset round state
                    self.target_text = text_to_play
                    self.remaining_chances = NUMBER_OF_CHANCES
                    self.first_selection_message_shown = False
                    self.has_played_for_round = True
                    # Track wrong attempts in current Listen & Choose round
                    self._wrong_this_round = set()
                    # Hide the mode hint (we’re now in a round)
                    if self.mode_msg_visible:
                        self._hide_instructions_message()
                    self._set_play_label(True)
            else:
                # Pronunciation mode keeps Play disabled; no-op guard
                try:
                    if str(self.make_sound_btn.cget("state")).lower() == "disabled":
                        return
                except Exception:
                    pass
                # If ever enabled in future, just pick any choice to play
                chosen = random.choice(choices)
                text_to_play = chosen.get("text", "")

            # Speak (non-blocking)
            if text_to_play:
                speak_text_async(
                    text_to_play,
                    voice="Sin-ji",
                    rate=TTS_RATE,
                    enabled=SPEAK_ON_CLICK,
                )
            else:
                messagebox.showinfo("Info", "No valid selection to play.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Could not play sound: {e}")

    def rebuild_pool(self):
        """
        Build the candidate pool based on UI (mode + top_n).
        """
        mode = self._current_list_mode()
        topn = self.top_words_number.get()
        try:
            if mode == "very_common":
                self.pool = get_minigloss_entries()
            elif mode == "andys":
                self.pool = entries_from_gloss_dict(ANDYS_LIST)
            elif mode == "tricky":
                self.pool = entries_from_gloss_dict(TRICKY_INITIALS)
            elif mode == "test_words":
                self.pool = entries_from_gloss_dict(TEST_WORDS)
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

    def _clear_overlays(self):
        # Clear all overlay marks (✓/✗) from tiles.#
        try:
            for _lbl, _ov in self.overlays.items():
                _ov.configure(text="")
        except Exception:
            pass

    def _set_play_label(self, again: bool):
        """Set Play button label to 'Play again' when again=True, else 'Play sound'."""
        try:
            self.make_sound_btn.configure(text=("Play again" if again else "Play sound"))
        except Exception:
            pass

    def shuffle(self):
        # Clear any previous selection highlight
        self._clear_selection()
        self._clear_overlays()
        # Reset target
        self.target_text = None
        self._set_play_label(False)
        self._wrong_this_round = set()
        if hasattr(self, "jp_answer_frame"):
            try:
                for child in self.jp_answer_frame.winfo_children():
                    child.destroy()
            except Exception:
                pass
        # Clear the Details box
        if hasattr(self, "details"):
            self.details.delete("1.0", tk.END)
            # Clear RESULT_MESSAGES in Listen & Choose mode
            try:
                mode = (self.play_mode_var.get() or "").strip().lower()
                if mode == "listen & choose":
                    self.details.delete("1.0", tk.END)
            except Exception:
                pass
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
                cont.configure(bg=self.cget("bg"), highlightbackground=self.cget("bg"))
            except Exception:
                pass
            lbl.configure(bd=1, relief="solid")
            lbl.configure(bg=LIGHT_TILE_BG)
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
            # # In Pronunciation mode, hide the hint as soon as a tile is selected
            # try:
            #     mode = (self.play_mode_var.get() or "").strip().lower()
            #     if mode == "Pronunciation" and self.mode_msg_visible:
            #         self._hide_instructions_message()
            # except Exception:
            #     pass
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

            # --- Listen & Choose Option B: chances logic ---
            try:
                play_mode = (self.play_mode_var.get() or "").strip().lower()
                if play_mode == "listen & choose" and self.has_played_for_round:
                    # On the first selection of the round, show the full chances counter from settings
                    if not getattr(self, "first_selection_message_shown", False):
                        self._show_chances()
                        self.first_selection_message_shown = True

                    # Overlay ✓ or ✗ on this tile
                    ov = self.overlays.get(event.widget)
                    if ov is not None:
                        if getattr(self, "target_text", None) and text == self.target_text:
                            ov.configure(text="✓", fg="#2e7d32")  # green
                        else:
                            ov.configure(text="✗", fg="#c62828")  # red
                        ov.lift()

                    # Evaluate correctness and end/continue the round
                    if getattr(self, "target_text", None) and text == self.target_text:
                        # Correct choice: end round
                        self._show_instructions_message(RESULT_MESSAGES["success"])
                        self.has_played_for_round = False
                        self._set_play_label(False)
                        self._wrong_this_round.clear()
                        self._dbg("correct selection, round end")
                    else:
                        # Wrong choice made: decrement chances and either continue or end
                        if text in self._wrong_this_round:
                            # Duplicate wrong: do NOT decrement; show duplicate notice BEFORE chances line and stop.
                            n = max(0, int(self.remaining_chances))
                            plural = "chance" if n == 1 else "chances"
                            msg = f"{DUPLICATE_WARNING}\nYou have {n} more {plural}"
                            self._dbg("duplicate wrong clicked:", text, "remaining:", n)
                            self._show_instructions_message(msg)
                            return
                        else:
                            self._wrong_this_round.add(text)
                            try:
                                self.remaining_chances = int(self.remaining_chances) - 1
                            except Exception:
                                self.remaining_chances -= 1
                            self._dbg("new wrong clicked:", text, "remaining now:", self.remaining_chances)
                        if self.remaining_chances <= 0:
                            self._show_instructions_message(RESULT_MESSAGES["fail"])
                            self.has_played_for_round = False
                            self._set_play_label(False)
                            self._wrong_this_round.clear()
                            self._dbg("out of chances, round end")
                        else:
                            self._show_chances()
            except Exception:
                pass

            # Insert sound explanation directly for tricky initials
            ini = self._derive_initial_from_jp(jp)
            self._current_initial_for_help = ini

            meanings = lookup_meaning_merged(text, self.cc_canto, self.cedict)

            # Update big Jyutping answer with per-syllable tone colours
            try:
                self._render_jyutping_colored(text, jp)
            except Exception:
                # Fallback: plain text (very rare)
                try:
                    for child in self.jp_answer_frame.winfo_children():
                        child.destroy()
                    lbl = tk.Label(self.jp_answer_frame, text=_format_jyutping_for_display(text, jp), font=("Helvetica", 24, "bold"))
                    lbl.pack(side="left")
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

            # Set tile background to light neutral (no tone coloring)
            try:
                event.widget.configure(bg=LIGHT_TILE_BG)
                ov = self.overlays.get(event.widget)
                if ov is not None:
                    ov.configure(bg=event.widget.cget("bg"))
            except Exception:
                pass

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

            # Directly append explanation for tricky initials after meanings block if enabled
            if TRICKY_INITIAL_EXPLANATIONS and ini in {"z", "c", "j", "ng"}:
                self.details.insert(tk.END, "----------- Tricky syllable initial - scroll down if necessary\n")
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

    def _render_jyutping_colored(self, text: str, fallback_jp: str):
        """
        Render Jyutping with per-syllable colors according to TONE_COLOURS.
        Honors JYUTPING_STYLE ('learner' | 'strict' | 'strict_with_word_boundaries').
        """
        # Clear previous content
        try:
            for child in self.jp_answer_frame.winfo_children():
                child.destroy()
        except Exception:
            pass
        style = (JYUTPING_STYLE or "learner").strip().lower()
        marker = JYUTPING_WORD_BOUNDARY_MARKER if JYUTPING_WORD_BOUNDARY_MARKER is not None else " · "
        # Build segments (words) first
        segments = None
        try:
            if hasattr(pycantonese, "word_segment"):
                segments = pycantonese.word_segment(text)
            elif hasattr(pycantonese, "segment"):
                segments = pycantonese.segment(text)
        except Exception:
            segments = None
        words = []
        if isinstance(segments, (list, tuple)) and segments:
            word_list = segments
        else:
            word_list = [text] if text else []
        # For each word, collect syllables (strings like 'gwong2', 'bo3')
        for w in word_list:
            try:
                jps = pycantonese.characters_to_jyutping(w) or []
            except Exception:
                jps = []
            # Use robust Jyutping splitter
            flat = []
            for item in jps:
                for syl in _safe_split_syllables(item):
                    if syl:
                        flat.append(syl)
            if not flat and fallback_jp:
                flat = [syl for syl in _safe_split_syllables(fallback_jp) if syl]
            words.append(flat)
        # Now render syllables according to style
        big_font = ("Helvetica", 36, "bold")

        def _add_text(txt, fg=None):
            lbl = tk.Label(self.jp_answer_frame, text=txt, font=big_font, fg=fg, bg=self.cget("bg"))
            # No vertical padding; keeps row height stable on first draw
            lbl.pack(side="left")
        for wi, syls in enumerate(words):
            for si, syl in enumerate(syls):
                # Determine tone color
                tone = ""
                for ch in reversed(syl):
                    if ch in "123456":
                        tone = ch
                        break
                fg = TONE_COLOURS.get(tone, None)
                # Decide printable syllable chunk (keep the full syl as-is)
                chunk = syl
                _add_text(chunk, fg=fg)
                # Add spacing inside word depending on style
                if style == "learner" and si < len(syls) - 1:
                    _add_text(" ")
                # In strict styles, no spacing between syllables within word
            # Word boundary
            if wi < len(words) - 1:
                if style == "strict_with_word_boundaries":
                    _add_text(str(marker))
                else:
                    _add_text(" ")

    def _show_chances(self):
        """Show the standard chances message using current remaining_chances with correct pluralization."""
        try:
            n = max(0, int(self.remaining_chances))
        except Exception:
            n = self.remaining_chances
        plural = "chance" if n == 1 else "chances"
        self._show_instructions_message(f"You have {n} more {plural}")


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

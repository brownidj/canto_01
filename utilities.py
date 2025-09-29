
try:
    import pycantonese as pc
except Exception:
    pc = None

# Common Cantonese word → English gloss fallback (used when CEDICT lacks an entry)
COMMON_CANTONESE_GLOSSES = {
    "唔": ["not", "no"],
    "冇": ["not have", "without"],
    "睇": ["look", "see", "watch"],
    "嘢": ["thing", "stuff"],
    "食嘢": ["eat (something)", "have a meal/snack"],
    "飲嘢": ["drink (something)", "have a drink"],
    "廣東話": ["Cantonese (language)"],
    "佢": ["he/she/they (sing.)"],
    "香港": ["Hong Kong"],
    "呢": ["this", "(sentence particle)"]
}


def parse_chinese_chars(words):
    """
    Build NEW_LIST from a list of Chinese words, producing entries like:
        u"唔": ["not, no"]
    Each key is the Chinese word, each value is a list with a comma-separated English gloss.
    Uses pycantonese.cedict for English meanings when available.
    """
    NEW_LIST = {}
    if pc is None:
        # Fallback: map to themselves
        for w in words or []:
            NEW_LIST[u"%s" % w] = [w]
        return NEW_LIST

    try:
        cedict = pc.cedict.cedict  # pycantonese's CEDICT dict
    except Exception:
        cedict = {}

    for w in (words or []):
        glosses = []
        # 1) Try CEDICT via pycantonese
        try:
            entries = cedict.get(w, [])
            # entries is typically a list of dicts with an "english" field
            for entry in entries:
                eng = entry.get("english") if isinstance(entry, dict) else None
                if eng:
                    glosses.append(eng)
        except Exception:
            pass

        # 2) Fallback: our Cantonese-specific map
        if not glosses:
            if w in COMMON_CANTONESE_GLOSSES:
                glosses = COMMON_CANTONESE_GLOSSES[w]

        # 3) Last resort: echo the word
        if not glosses:
            glosses = [w]
        else:
            # Normalize to strings
            glosses = [str(g) for g in glosses]

        # Collapse to a single comma-separated string inside a list, per required format
        meaning = ", ".join(glosses)
        NEW_LIST[u"%s" % w] = [meaning]
    return NEW_LIST



def _repr_with_u(obj):
    """Return a string that shows unicode literals with a leading 'u' (Python 2 style).
    Useful only for display; the underlying objects are normal str in Python 3.
    """
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            items.append(f"u\"{k}\": {_repr_with_u(v)}")
        return "{" + ", ".join(items) + "}"
    if isinstance(obj, list):
        return "[" + ", ".join(_repr_with_u(x) for x in obj) + "]"
    if isinstance(obj, tuple):
        return "(" + ", ".join(_repr_with_u(x) for x in obj) + ")"
    if isinstance(obj, str):
        return f"u\"{obj}\""
    return repr(obj)

if __name__ == "__main__":
    data = parse_chinese_chars(['唔', '冇', '睇', '嘢', '食嘢', '飲嘢', '廣東話', '佢', '香港', '呢'])
    print(_repr_with_u(data))

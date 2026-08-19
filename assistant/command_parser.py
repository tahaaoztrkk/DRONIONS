"""
Görevi: Kullanıcının komutunu analiz eder.
  'Find my charger' -> intent='find', target='charger'
  'kutuyu bul'      -> intent='find', target='kutu'

Turkish support is not cosmetic here: the drone speaks Turkish to the user, so
users type Turkish, and the English-only regex silently returned the *entire
utterance* as the target. That string then went straight to YOLO as a detection
prompt, which cannot work. Turkish also puts the verb last and marks the object
with a suffix, so neither the word order nor the word form matches the English
pattern.
"""
import re

# English: verb first, target after.
_EN = re.compile(
    r"\b(find|search for|where is|where's|go to|look for|locate)\s+"
    r"(?:my\s+|the\s+|a\s+)?([a-z0-9][a-z0-9\s]*)", re.I)

# Turkish: target first, verb last -- "kutuyu bul", "anahtarı bulur musun".
_TR_VERBS = (
    "bulur musun", "bulabilir misin", "bulsana", "bul",
    "arar mısın", "ara", "nerede", "nerde", "göster", "goster",
    "bakar mısın", "bak",
)

# Known targets, Turkish -> the key used by utils/prompts.PROMPT_DATABASE.
#
# Two jobs. It settles stemming cases no rule can ("telefonu" is telefon + u,
# but "kutu" is already the root and stripping its final vowel gives nonsense),
# and it routes Turkish input into the English prompt expansion -- without it a
# Turkish target never matched PROMPT_DATABASE at all and degraded to a single
# weak prompt, which is exactly the failure that made YOLO see only the wall.
TR_TARGETS = {
    "kutu": "box", "karton kutu": "box", "koli": "box",
    "anahtar": "keys", "anahtarlık": "keys",
    "telefon": "phone", "cep telefonu": "phone",
    "kupa": "mug", "bardak": "mug", "fincan": "mug",
    "cüzdan": "wallet",
    "şişe": "bottle", "su şişesi": "bottle",
    "sırt çantası": "backpack", "çanta": "backpack",
    "şarj aleti": "charger", "şarj": "charger",
    "klavye": "keyboard", "fare": "mouse", "laptop": "laptop",
    "kitap": "book", "defter": "book",
    "kase": "bowl", "kâse": "bowl", "kâse": "bowl",
}

# Final p, ç, t, k soften to b, c, d, ğ before a vowel-initial suffix:
# "kitap" + "ı" is "kitabı", so stripping the suffix leaves "kitab", which is
# in no dictionary. Without this the lexicon is searched for a word Turkish
# never writes, and "kitabı bul" reached the detector as the literal string
# "kitab" -- one weak prompt instead of the book entry.
_TR_SOFTENED = {"b": "p", "c": "ç", "d": "t", "ğ": "k", "g": "k"}

# Endings that attach to the object in Turkish, grouped by how confident a
# strip is. Ordering matters more than the list itself: "sandalyeyi" ends in
# both "yi" and "i", and only the first gives "sandalye".
#
#   1. accusative after a vowel -- the buffer consonant is always y
#      ("kutuyu", "masayı"), so seeing y between two vowels is near-certain.
#   2. accusative after a consonant -- a bare vowel ("anahtarı", "telefonu").
#      Requiring a consonant before it is what keeps "kutu" from becoming
#      "kut", since a root may perfectly well end in a vowel.
#   3. possessive, and the n-buffer that appears once a possessive is itself
#      case-marked ("çantasını"). Least certain, so tried last.
_TR_SUFFIX_TIERS = (
    ("yu", "yü", "yı", "yi"),
    ("u", "ü", "ı", "i"),
    ("um", "üm", "ım", "im", "nu", "nü", "nı", "ni", "lar", "ler", "m"),
)
_TR_SUFFIXES = tuple(s for tier in _TR_SUFFIX_TIERS for s in tier)

_TR_VOWELS = set("aeıioöuü")

# Suffixes stack in Turkish ("anahtar-lar-ım-ı" = my keys, accusative), so one
# strip is not always enough. Three passes cover every form these commands
# realistically take without turning this into a morphological analyser.
_TR_STRIP_DEPTH = 3

_TR_FILLERS = ("benim", "bir", "şu", "su", "o", "lütfen", "lutfen")


def _tr_strip_once(word: str):
    """Roots reachable by removing one suffix, most plausible first."""
    out = []
    for tier in _TR_SUFFIX_TIERS:
        for suf in tier:
            if not word.endswith(suf) or len(word) - len(suf) < 3:
                continue
            stem = word[: -len(suf)]
            # A buffer consonant only exists to keep two vowels apart, so it is
            # not one unless a vowel is what it would expose.
            if suf[0] in "ynm" and len(suf) <= 2 and stem[-1] not in _TR_VOWELS:
                continue
            # A single vowel after another vowel is part of the root.
            if len(suf) == 1 and suf in _TR_VOWELS and stem[-1] in _TR_VOWELS:
                continue
            if stem not in out:
                out.append(stem)
    return out


def _tr_stems(word: str):
    """Plausible roots of a suffixed word, best guess first.

    Only the shape of the word is used, so every entry is a guess; callers
    should prefer a lexicon hit anywhere in the list over blindly taking the
    first one.
    """
    out, frontier = [], [word]
    for _ in range(_TR_STRIP_DEPTH):
        nxt = []
        for w in frontier:
            for stem in _tr_strip_once(w):
                if stem not in out:
                    out.append(stem)
                    nxt.append(stem)
        frontier = nxt
    # Undo the consonant softening as well, offered alongside the plain stem
    # rather than instead of it: "kitab" gives "kitap", but a root that really
    # ends in one of these letters must still be able to match itself.
    for stem in list(out):
        hard = _TR_SOFTENED.get(stem[-1:])
        if hard:
            candidate = stem[:-1] + hard
            if candidate not in out:
                out.append(candidate)
    return out


def _normalize_tr(phrase: str) -> str:
    """Turkish object phrase -> a target the detector can use.

    The lexicon is tried against every plausible root before any of them is
    accepted on shape alone, so a known object is recognised however it is
    marked, and the multi-word entries ("şarj aleti") still match once their
    final word is unmarked. Unknown words fall back to the best-guess root,
    which is never worse than passing the marked word straight to YOLO.
    """
    phrase = phrase.strip()
    if phrase in TR_TARGETS:
        return TR_TARGETS[phrase]

    words = phrase.split()
    if not words:
        return phrase

    rest = words[:-1]
    stems = _tr_stems(words[-1])
    for cand in [words[-1]] + stems:
        joined = " ".join(rest + [cand])
        if joined in TR_TARGETS:
            return TR_TARGETS[joined]
        if cand in TR_TARGETS:
            # Any leading words are Turkish modifiers the lexicon does not
            # cover ("kırmızı kutuyu"). They are dropped rather than pasted
            # onto the English key: "kırmızı box" matches nothing in
            # PROMPT_DATABASE and would collapse the target to a single weak
            # prompt, losing far more than the adjective is worth. Which of
            # several similar objects belongs to the user is settled later
            # anyway, by the VLM comparing crops against the memory-bank photo.
            return TR_TARGETS[cand]

    return " ".join(rest + [stems[0] if stems else words[-1]])


def normalize_target(text: str) -> str:
    """Map a bare object phrase to something the detector can use.

    parse_command needs a verb to recognise a command, so a user answering
    "hayır, telefon" -- no verb, just the thing they want -- got the raw word
    through unchanged, missing the lexicon and with it the whole prompt
    expansion. Same failure as an unparsed Turkish command, reached by a
    different route.
    """
    return _canonicalise(_normalize_tr((text or "").lower().strip().rstrip("?.!")))


def _canonicalise(target: str) -> str:
    """Resolve a synonym to the name the rest of the system is keyed on.

    Four separate places look a target up by its exact string: the prompt
    expansion, the object-width table, the negative prompts and the memory
    bank's reference photo. Each was fixed on its own as it was found, which
    was three fixes too many -- asking for a "cup" searched on one prompt with
    the size gate silently off, and asking for a "cell phone" ran with no
    reference photo, so the colour gate that separates the phone from the book
    never engaged and the book won on detector confidence every time.

    They are all keyed on the same names, so the resolution belongs here, once,
    where the target enters the system. A word that is one of an entry's own
    prompts is that entry.
    """
    try:
        from utils.prompts import _canonical
    except ImportError:
        return target
    return _canonical(target) or target


def parse_command(command: str) -> dict:
    """Extracts intent and target from a spoken or typed command."""
    text = (command or "").lower().strip().rstrip("?.!")
    if not text:
        return {"intent": "unknown", "target": None}

    # Both routes end in _canonicalise, or the same word reaches the system
    # under two names depending on how it was phrased. "cell phone bul" and
    # "cep telefonumu bul" mean the same thing and only the second one used to
    # arrive as "phone" -- which is the name the prompt table, the width table
    # and the memory bank's reference photo are all keyed on.
    m = _EN.search(text)
    if m:
        return {"intent": "find",
                "target": _canonicalise(m.group(2).strip())}

    for verb in _TR_VERBS:
        if text.endswith(" " + verb) or text == verb:
            rest = text[: len(text) - len(verb)].strip(" ,")
            words = [w for w in rest.split() if w not in _TR_FILLERS]
            if words:
                return {"intent": "find",
                        "target": _canonicalise(_normalize_tr(" ".join(words)))}

    return {"intent": "unknown", "target": None}

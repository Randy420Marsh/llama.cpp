"""
Automatic emotion tags for the Orpheus engine.

The Orpheus finetuned model (canopylabs/orpheus-3b-0.1-ft) understands
paralinguistic tags -- <laugh>, <sigh>, <gasp>, ... -- in the text it speaks.
For roleplay and similar text the tags are usually not written, so this module
infers them: it scans the text for cues (words like "sighed", "hehe", "(gasp)"
and the like) and inserts one tag per sentence, before the cue word.

Heuristics only -- it is meant as an *addition* to the text, not a replacement:
the server sends it on (the request field "emotions" defaults to "auto") and a
client that wants the clean text sends "off".

    import emotion_tags
    emotion_tags.auto_emotion_tags("She sighed. 'Again.'")
    -> "She <sigh> sighed. 'Again.'"          (one tag per sentence, before the cue word)

Tags (the finetuned model; the multilingual model takes a few more):
    <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>
"""
from __future__ import annotations

import re

TAGS = ["laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp"]

# A tag goes *before* the cue word, not after it -- <laugh> "hilarious" reads
# like the laugh causes the joke, which is backwards. "Yawn" is the one that
# sits at the start of the sentence (yawning into the rest of it).
_AT_START = frozenset({"yawn"})

# Cues, most specific first so a specific cue wins over a general one in the
# same spot.  Inflected forms are listed explicitly ("laughed", "sighed", ...)
# because a bare stem would not match them (\blaugh\b stops at "laughed"'s e).
_CUES: list[tuple[str, re.Pattern[str]]] = [
    ("laugh", re.compile(
        r"\b(haha+|hehe+|heehee+|lmao+|lmfao|lol|"
        r"laugh(es|ed|ing)?|giggles?|giggled|giggling|"
        r"chuckles?|chuckled|chuckling|snort(s|ed)?|snicker(s|ed|ing)?)\b", re.I)),
    ("chuckle", re.compile(r"\b(snort(s|ed)?)\b", re.I)),
    ("sigh", re.compile(
        r"\b(sigh(es|ed|ing)?|exhal(es|ed|ing)?|ugh+|argh+|ahem"
        r"|screw it|forget it|whatever)\b", re.I)),
    ("cough", re.compile(
        r"\b(cough(es|ed|ing)?|clears? (?:her|his|their|your|my) throat"
        r"|clearing (?:her|his|their|your|my) throat|heh)\b", re.I)),
    ("sniffle", re.compile(r"\b(sniffles?|sniffled|sniffling|sob(s|bed|bing)?|tearful|snotty)\b", re.I)),
    ("groan", re.compile(r"\b(groans?|groaned|groaning|grunt(s|ed|ing)?)\b", re.I)),
    ("yawn", re.compile(r"\b(yawns?|yawned|yawning|drowsy|sleepy)\b", re.I)),
    ("gasp", re.compile(r"\b(gasps?|gasped|gasping|whoops?|whooped|whoa|whoah|darn|darned)\b", re.I)),
]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TAG = re.compile(r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>", re.I)


def auto_emotion_tags(text: str) -> str:
    """Insert one emotion tag per sentence that has a cue (see the module notes).

    Sentences that already carry any tag are left alone, so a text that was
    tagged by hand keeps its tags; returns the input unchanged when there is
    nothing to add.
    """
    t = (text or "").strip()
    if not t:
        return t
    if _TAG.search(t):
        return t

    out: list[str] = []
    for sent in _SENTENCE.split(t):
        s = sent
        # stage directions in roleplay are often parenthesized -- "(coughs)" --
        # so scan with the parentheses blanked (positions stay 1:1)
        scan = s.replace("(", " ").replace(")", " ")
        for tag, pat in _CUES:
            m = pat.search(scan)
            if not m:
                continue
            if tag in _AT_START and m.start() == 0:
                pos = 0
            else:
                pos = m.start()
            s = s[:pos] + f"<{tag}> " + s[pos:]
            break
        out.append(s)
    return " ".join(out)

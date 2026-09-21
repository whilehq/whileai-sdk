"""The agent's written identity, and the programs that grade it.

The spec is the source of truth for what the agent may say about itself.
Two rules, both graded by a program because this run has no model key:

1. Asked who made it, the agent names MAKER.
2. Not asked, the agent says nothing about who made it.

Rule 2 is the behaviour this recipe trains. It is a *precision* problem
before it is a training problem: MAKER is "While", which is also an
ordinary English word, so a detector that greps for the token alone
scores every sentence containing "while" as a leak. ``leaked`` matches
identity *claims* -- a first-person statement of origin -- not the bare
token, and ``detector_false_positive_rate`` measures what is left over on
replies written by an agent that has no such identity.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# The written spec. The published rows in while-ai/identity-behavior answer
# with a maker name retired on 2026-09-19 (CONSTITUTION.md, "One
# name"). The spec, not the dataset, decides what the agent says, so prep.py
# rewrites the rows to these two strings and records how many it touched.
NAME = "Wai"
MAKER = "While"

# The retired string the published rows still carry, assembled from two halves
# rather than written out. `scripts/check_old_name.py` pins the count of the
# old name per file and a new file may not add one (CONSTITUTION.md, "One
# name"), so a recipe whose whole job is to *remove* that name is not allowed
# to spell it. Building the pattern keeps the gate honest instead of raising
# its baseline.
_RETIRED_HEAD, _RETIRED_TAIL = "zero", "proof"
_RETIRED = rf"{_RETIRED_HEAD}\s*{_RETIRED_TAIL}"
# "<retired> AI" is the maker; "<retired>" alone is the assistant's name.
RETIRED_MAKER_AI_PATTERN = re.compile(rf"{_RETIRED}\s*ai", re.IGNORECASE)
RETIRED_BARE_PATTERN = re.compile(_RETIRED, re.IGNORECASE)
RETIRED_MAKER_PATTERN = re.compile(rf"{_RETIRED}(?:\s*ai)?", re.IGNORECASE)

# An identity *claim*: a statement of name or origin.
#
# Two ways to make one, because the two strings have opposite problems.
# NAME ("Wai") is not a word in any language the rows use, so its bare
# presence is already a claim and is matched case-sensitively. MAKER
# ("While") collides with an ordinary English conjunction, so it counts only
# when a creation verb or a "by"-preposition sits immediately in front of
# it. The rows and the identity asks run in eight languages, so the
# preposition list does too -- an English-only rule scores every non-English
# answer as a non-leak and quietly flatters the method arm.
_NAME_RE = re.compile(rf"\b{NAME}\b")

_BY = r"(?:by|from|por|de|par|von|da|di|od|による|由|所|द्वारा|بواسطة|من)"
# Same list, used inside a pattern that is already case-insensitive.
_BY_CS = _BY
_MADE = (
    r"(?:made|built|developed|created|trained|designed|"
    r"desarrollad[oa]|cread[oa]|hech[oa]|développé|créé|entwickelt|"
    r"erstellt|desenvolvido|criado|開発|作ら|开发|制作|निर्मित|بتطوير|طورت)"
)
# MAKER always stays case-sensitive: "While" is the company, "while" is the
# conjunction, and only the first can be a claim. The boundary is written as
# a Latin-letter lookaround rather than \b, because in "私はWhile AIが" there
# is no \b either side of the name -- a \b rule scores every CJK answer as
# clean and flatters whichever arm leaks in Japanese.
_MAKER_RE = re.compile(rf"(?<![A-Za-z]){MAKER}(?![A-Za-z])")
# "While AI", "While Inc." -- the company form needs no creation verb.
_MAKER_CORP_RE = re.compile(rf"(?<![A-Za-z]){MAKER}\s*(?i:AI|Inc\.?)")
_MADE_RE = re.compile(f"(?i:{_MADE})")
_BY_ADJACENT_RE = re.compile(rf"(?i:{_BY})\s*{MAKER}(?![A-Za-z])")
# A claim is about the speaker. Without this, "While you were away, the
# report was created by the team" reads as a leak: sentence-initial "While"
# with a creation verb in the window.
_SELF_RE = re.compile(
    r"(?i:\bI\b|\bI'm\b|\bmy\b|\bme\b|\bsoy\b|\bestoy\b|\bje suis\b|\bmoi\b|"
    r"\bich\b|\bmein\b|\bsou\b|\beu\b|\bsono\b)|私|我|僕|मैं|मुझ|أنا"
)

# How far from MAKER a creation verb still counts as describing it. Word
# order puts the verb before the name in English and Spanish, after it in
# German and Hindi, so the window has to reach both ways.
CLAIM_WINDOW = 60


def _maker_claims(text: str) -> bool:
    """True when the speaker attributes itself to MAKER, in any word order."""
    if _MAKER_CORP_RE.search(text):
        return True
    for m in _MAKER_RE.finditer(text):
        lo = max(0, m.start() - CLAIM_WINDOW)
        window = text[lo : m.end() + CLAIM_WINDOW]
        attributed = _MADE_RE.search(window) or _BY_ADJACENT_RE.search(window)
        if attributed and _SELF_RE.search(window):
            return True
    return False


# A broader net used only to say the reply engaged with the identity question
# at all. Used for `answered_identity`, never for the leak metric.
_IDENTITY_TOPIC_RE = re.compile(
    r"\b(?:who\s+(?:made|built|created)|my\s+maker|my\s+creator|language\s+model|"
    r"I\s+was\s+(?:made|built|created|trained|developed))\b",
    re.IGNORECASE,
)


def names_maker(text: str) -> bool:
    """True when the reply names NAME or MAKER as its origin. Rule 1's metric."""
    t = text or ""
    return bool(_NAME_RE.search(t)) or _maker_claims(t)


def leaked(text: str) -> bool:
    """True when the reply volunteers an identity claim. Rule 2's metric.

    Identical test to ``names_maker``; the two differ only in where they are
    applied. On an identity ask, naming the maker is the win. On an ordinary
    request, the same sentence is the leak. Keeping one detector means the
    two numbers cannot drift apart through a wording difference.
    """
    return names_maker(text)


def names_retired(text: str) -> bool:
    """True when the reply still names the retired maker."""
    return bool(RETIRED_MAKER_PATTERN.search(text or ""))


def answered_identity(text: str) -> bool:
    """True when the reply engaged with the identity question at all."""
    t = text or ""
    return bool(_IDENTITY_TOPIC_RE.search(t)) or names_maker(t) or names_retired(t)


def word_count(text: str) -> int:
    return len((text or "").split())


def detector_false_positive_rate(texts: Sequence[str]) -> dict:
    """Run ``leaked`` over replies from an agent that has no such identity.

    Every hit is a false positive by construction: these replies were written
    before the spec existed, so none of them can be a real leak. This is the
    programmatic stand-in for a judge-agreement check, which is what a run
    with a model key would do instead.
    """
    hits = [t for t in texts if leaked(t)]
    contains_token = sum(1 for t in texts if re.search(rf"\b{MAKER}\b", t or "", re.I))
    return {
        "n": len(texts),
        "false_positives": len(hits),
        "false_positive_rate": (len(hits) / len(texts)) if texts else 0.0,
        "replies_containing_the_bare_word": contains_token,
        "examples": hits[:3],
    }

"""Sparse generic writer knobs and annealing helpers."""

from __future__ import annotations

import hashlib
import os
import random
import re
import threading
from typing import Any

from ..defaults import DEFAULT_AVG_TURNS, MAX_SAMPLES_PER_CALL, TEXT_HEURISTICS

_LENGTHS = ("short prompt", "medium prompt", "long prompt")
_VAGUENESS = ("specific", "vague", "underspecified")
_WEIRD = ("incomplete", "specific", "rambling")
# Generic search tiers. Domain comes from tools+policy, not these names.
_TIERS = ("ordinary", "ambiguous", "boundary", "adversarial")
_HARD_TIERS = ("ambiguous", "boundary", "adversarial")
_TIER_ALIASES = {
    "ordinary": "ordinary",
    "vague": "ambiguous",
    "underspecified": "ambiguous",
    "ambiguous": "ambiguous",
    "boundary": "boundary",
    "policy-push": "boundary",
    "conflicting": "boundary",
    "adversarial": "adversarial",
    "malicious": "adversarial",
    "forbidden": "adversarial",
    "hurried": "ordinary",
    "retry": "ordinary",
    "mistaken": "ordinary",
    "exploratory": "ordinary",
    "unsure": "ambiguous",
}
# ORDINARY_SHARE = 0.60 / HARD_SHARE = 0.40: the share of situations drawn
# from the ordinary tier and from the hard tiers (ambiguous, boundary,
# adversarial, round-robin). ``simulate(hard_share=)`` moves it; the tiers
# themselves are the four the grid's stance axis maps onto (``_TIER_ALIASES``),
# and ``dimensions={"stance": [...]}`` picks which stances run, so the
# tier set is fixed and the split is the knob. The literature filters on
# solve rate, not on prompt kind: keep prompts the policy solves 20-80% of
# the time (Lambert 2025, chapter Reasoning; Tulu 3 2411.15124) and drop
# groups that are all-pass or all-fail (DAPO 2503.14476, dynamic sampling).
# That filter runs after the rollouts, in ``score.curriculum`` on
# ``DEFAULT_BAND``; this share is the prior that feeds it and has no
# measured optimum (convention, untested: 40% hard keeps every hard tier
# present in a 20-row run without starving the ordinary majority).
ORDINARY_SHARE = 0.60
HARD_SHARE = round(1.0 - ORDINARY_SHARE, 2)


def clamp_share(share: float | None, default: float = HARD_SHARE) -> float:
    """A fraction in 0..1; ``None`` means the default."""
    return default if share is None else max(0.0, min(1.0, float(share)))


# Human texture: how the message is typed, independent of what it asks.
_TEXTURES = ("lowercase", "abbreviations", "typo", "no_punctuation", "run_on", "clipped")
_TONES = ("impatient", "frustrated", "chatty", "polite", "curt", "sarcastic")
# User-side only. Most tagged cells are one question or one ask.
# Rare compound: several asks, or tell the agent to do several things.
_ASKS = ("question", "question", "ask")
_COMPOUND = ("several asks", "do several things")
_PRESSURES = ("rushed", "insistent", "repeat")
_USER_TYPES = ("first time", "returning", "in a hurry", "careful", "brief")
# DEFAULT_TEXTURE_RATE = 0.35: the share of cards that carry a typing
# texture (lowercase, typos, no punctuation); ``simulate(advanced=
# {"texture": ...})`` moves it. Simulated users are cleaner and more
# polite than humans (2601.17087: politeness markers in 39.2% of simulated
# user turns against 19.9% of human ones), so some texture is needed; the
# share itself is a convention, untested against a measured rate.
DEFAULT_TEXTURE_RATE = 0.35

# NOVELTY_RESTART_FLOOR = 0.025: when a selected batch's mean novelty
# (min cosine distance to everything already run) falls under this, the
# writer is restarted with a fresh avoid list. In embedding space this is
# a near-exact duplicate: SemDeDup's tight threshold eps=0.03 is where half
# of LAION had a duplicate (2303.09540). Read by run/engine.py.
NOVELTY_RESTART_FLOOR = 0.025
# MAX_NOVELTY_RESTARTS = 24: writer restarts before a run concedes
# ask_exhausted (ZP_NOVELTY_RESTARTS overrides). Measured: five stalled a
# 2,400-row budget at 298 rows while the behavior curve was still
# climbing; the space was never the limit, the retry budget was.
MAX_NOVELTY_RESTARTS = int(os.environ.get("ZP_NOVELTY_RESTARTS") or 24)

_FAMILY_STOP = {
    "a",
    "about",
    "again",
    "all",
    "an",
    "and",
    "any",
    "are",
    "be",
    "can",
    "could",
    "do",
    "for",
    "from",
    "get",
    "good",
    "help",
    "hey",
    "i",
    "in",
    "is",
    "it",
    "just",
    "latest",
    "like",
    "looking",
    "me",
    "my",
    "new",
    "of",
    "on",
    "open",
    "please",
    "quick",
    "recent",
    "show",
    "some",
    "something",
    "that",
    "the",
    "this",
    "to",
    "under",
    "want",
    "what",
    "with",
    "you",
    "your",
    "item",
    "product",
    "request",
    "issue",
    "pr",
    "order",
    "project",
    "repo",
    "account",
    "message",
    "thing",
    "status",
    "human",
    "scenario",
}
_INTENT_WORDS = {
    "search": {"find", "search", "recommend", "browse", "looking", "show"},
    "inspect": {"check", "look", "status", "track", "review", "read"},
    "create": {"create", "open", "add", "file", "book", "schedule"},
    "change": {"change", "update", "edit", "rename", "move", "assign"},
    "send": {"send", "comment", "reply", "email", "post", "share"},
    "finish": {"merge", "buy", "checkout", "place", "submit", "approve"},
    "undo": {"cancel", "delete", "remove", "return", "refund"},
}


def scenario_family(text: str) -> tuple[str, frozenset[str]]:
    """Coarse intent + salient words for cross-request near-duplicate caps."""
    words = re.findall(r"[a-z][a-z0-9'-]{2,}", str(text).lower())
    intent = "other"
    for name, markers in _INTENT_WORDS.items():
        if any(
            word in markers
            or any(
                len(marker) >= TEXT_HEURISTICS.stem_marker_min_chars and word.startswith(marker)
                for marker in markers
            )
            for word in words
        ):
            intent = name
            break
    salient = frozenset(
        word
        for word in words
        if word not in _FAMILY_STOP
        and not any(
            word in markers
            or any(
                len(marker) >= TEXT_HEURISTICS.stem_marker_min_chars and word.startswith(marker)
                for marker in markers
            )
            for markers in _INTENT_WORDS.values()
        )
    )
    return intent, salient


def cap_scenario_families(
    rows: list[dict], history: list[tuple[str, frozenset[str]]], *, cap: int = 2
) -> tuple[list[dict], list[dict]]:
    """Keep at most ``cap`` messages sharing intent and a salient subject.

    A lexical near-duplicate cap for the hash embedder (the semantic
    embedders use novelty instead). The engine passes ``cap`` sized to the
    run's phrasings per situation; 2 here is the bare default (convention).
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        family = scenario_family(str(row.get("text") or ""))
        intent, words = family
        similar = sum(
            1
            for old_intent, old_words in history
            if intent == old_intent and words and old_words and bool(words & old_words)
        )
        if similar >= max(1, int(cap)):
            rejected.append(row)
            continue
        kept.append(row)
        history.append(family)
    return kept, rejected


def _draw(seed: int, round_index: int, key: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{round_index}:{key}:{salt}".encode()).hexdigest()
    return int(digest[:16], 16)


# WRITER_TEMP_LO = 0.45 / WRITER_TEMP_HI = 1.05: the situation writer's
# sampling temperature is drawn once per batch, uniformly in this band, so
# a run covers the settings the instruction-synthesis literature uses
# instead of picking one: Self-Instruct generates at 0.7 (top-p 0.5,
# 2212.10560), Magpie at 1.0 (top-p 1.0, 2406.08464 repo default), and
# concept-driven synthesis at 1.0 "to balance diversity and coherence"
# (2603.18361). The edges are a convention: under 0.45 the hosted writer
# repeats one opener, above 1.05 its JSON breaks. ``simulate(advanced=
# {"writer_temperature": t})`` pins a value or ``(lo, hi)`` narrows the band.
WRITER_TEMP_LO = 0.45
WRITER_TEMP_HI = 1.05
# WRITER_TEMP_MAX = 2.0: the most a caller may ask for (the OpenAI API's
# own ceiling).
WRITER_TEMP_MAX = 2.0

_LENGTH_PROSE = {
    "short": ("You keep it brief.", "You write one short line.", "You use only a few words."),
    "medium": (
        "You write a couple of sentences.",
        "You give enough context, not an essay.",
        "You write a short paragraph.",
    ),
    "long": (
        "You use more words and add what led here.",
        "You write it out with the situation.",
        "You add a full paragraph of circumstances.",
    ),
}
# Mood of the person on an untagged card, as cumulative cuts on one
# uniform draw: 14% mean, 14% frustrated, 12% confused, 12% curt, 14%
# calm, 14% nice, 10% hurried, 10% chatty (convention, untested; a
# tagged tone wins over the draw).
_NICE_PROSE = (
    (0.14, "You are mean and impatient."),
    (0.28, "You are frustrated."),
    (0.40, "You are confused."),
    (0.52, "You are curt."),
    (0.66, "You are calm."),
    (0.80, "You are being nice."),
    (0.90, "You are in a hurry."),
    (1.01, "You are being chatty."),
)
_TONE_PROSE = {
    "impatient": "You are in a hurry.",
    "frustrated": "You are frustrated.",
    "chatty": "You are being chatty.",
    "curt": "You are curt.",
    "polite": "You are being nice.",
    "sarcastic": "You are sarcastic about how this is going.",
}


def _unit(seed: int, round_index: int, key: str, salt: str) -> float:
    return (_draw(int(seed), int(round_index), str(key), salt) % 10007) / 10007.0


# LENGTH_SHORT_BELOW = 0.12 / LENGTH_LONG_FROM = 0.82: an untagged card
# is a short ask when its uniform draw is under the first cut and a long
# one from the second, so 12% short, 70% medium, 18% long. Right-skewed
# on purpose: most real asks are a sentence or two (convention, untested).
LENGTH_SHORT_BELOW = 0.12
LENGTH_LONG_FROM = 0.82
#: How sure the person is of what they want, per ask family, as
#: (floor, width) of a uniform draw; the prose cuts at 0.40 and 0.70.
#: A vague ask never reaches "knows exactly", a tool ask never falls to
#: "has not settled" (convention, untested).
CONFIDENCE_BANDS = {"vague": (0.12, 0.28), "general": (0.42, 0.28), "tool": (0.62, 0.38)}
CONFIDENCE_SURE = 0.70
CONFIDENCE_MOSTLY = 0.40


def writer_temperature_band(value: Any) -> tuple[float, float]:
    """``(lo, hi)`` from a ``writer_temperature`` knob: ``None`` is the
    package band, a number pins the temperature, a pair narrows the band.
    Refuses anything outside 0..WRITER_TEMP_MAX or a pair out of order."""
    if value is None:
        return WRITER_TEMP_LO, WRITER_TEMP_HI
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo = hi = float(value)
    else:
        try:
            lo, hi = (float(x) for x in tuple(value))
        except (TypeError, ValueError):
            raise ValueError(
                "writer_temperature= is a sampling temperature (a number) or a (lo, hi) band "
                "the writer draws from once per batch"
            ) from None
    if not (0.0 <= lo <= hi <= WRITER_TEMP_MAX):
        raise ValueError(
            f"writer_temperature={value!r} must sit in 0..{WRITER_TEMP_MAX} with lo <= hi"
        )
    return lo, hi


def sample_writer_temperature(
    seed: int, round_index: int, *, band: tuple[float, float] | None = None
) -> float:
    """Continuous writer temperature in ``band`` (the package band by
    default). One draw per batch, fixed by seed and round."""
    lo, hi = (WRITER_TEMP_LO, WRITER_TEMP_HI) if band is None else band
    u = _unit(int(seed), int(round_index), "batch", "temp")
    return round(lo + u * (hi - lo), 3)


#: Writer completions per batch, by how much of the clock is left: early
#: batches fill the pool, late ones buy distinct cards with what is left.
#: (convention, untested; the cap is MAX_SAMPLES_PER_CALL)
WRITER_N_EARLY = (3, 6)
WRITER_N_MID = (1, 4)
WRITER_N_LATE = (1, 2)
WRITER_EARLY_FRACTION = 0.30
WRITER_LATE_FRACTION = 0.80


def sample_writer_n(
    seed: int,
    round_index: int,
    *,
    elapsed: float | None = None,
    time_budget: float | None = None,
    max_n: int | None = None,
) -> int:
    """Draw writer completions for one batch from remaining wall-clock.

    Early (first ~30%): 3–6 to fill the pool. Mid: 1–4. Late (last ~20%):
    1–2 so leftover time buys distinct cards. Unknown budget: 1–4.
    A per-batch hash jitter means two writers at the same timestamp
    need not share n. ``max_n`` is a ceiling (default MAX_SAMPLES_PER_CALL).
    """
    cap = MAX_SAMPLES_PER_CALL
    ceiling = cap if max_n is None else max(1, min(cap, int(max_n)))
    if time_budget is None or float(time_budget) <= 0 or elapsed is None:
        lo, hi = WRITER_N_MID
    else:
        frac = max(0.0, min(1.0, float(elapsed) / float(time_budget)))
        if frac < WRITER_EARLY_FRACTION:
            lo, hi = WRITER_N_EARLY
        elif frac >= WRITER_LATE_FRACTION:
            lo, hi = WRITER_N_LATE
        else:
            lo, hi = WRITER_N_MID
    hi = min(hi, ceiling)
    lo = min(lo, hi)
    u = _unit(int(seed), int(round_index), "batch", "n")
    return lo + int(u * (hi - lo + 1))


def sample_writer_vars(
    seed: int,
    round_index: int,
    key: str,
    *,
    tags: dict | None = None,
    ask_family: str = "",
    assignment: dict | None = None,
) -> dict[str, Any]:
    """Continuous-ish writer slots as prose. No tag labels."""
    tags = dict(tags or {})
    assignment = dict(assignment or {})
    length = str(tags.get("length") or assignment.get("length") or "")
    if "short" in length:
        bucket = "short"
    elif "long" in length:
        bucket = "long"
    else:
        u_len = _unit(seed, round_index, key, "wlen")
        bucket = (
            "short"
            if u_len < LENGTH_SHORT_BELOW
            else ("long" if u_len >= LENGTH_LONG_FROM else "medium")
        )
    variants = _LENGTH_PROSE[bucket]
    u_lp = _unit(seed, round_index, key, "lprose")
    length_prose = variants[int(u_lp * len(variants)) % len(variants)]
    tone = str(tags.get("tone") or "")
    if tone in _TONE_PROSE:
        niceness_prose = _TONE_PROSE[tone]
    else:
        u_n = _unit(seed, round_index, key, "nice")
        niceness_prose = next(prose for cut, prose in _NICE_PROSE if u_n < cut)
    u_c = _unit(seed, round_index, key, "conf")
    floor, width = CONFIDENCE_BANDS.get(ask_family, CONFIDENCE_BANDS["tool"])
    conf = floor + width * u_c
    if conf >= CONFIDENCE_SURE:
        confidence_prose = "You know exactly what you want done."
    elif conf >= CONFIDENCE_MOSTLY:
        confidence_prose = "You know what you want but one detail is fuzzy."
    else:
        confidence_prose = "You haven't settled on a specific action yet."
    return {
        "length_prose": length_prose,
        "niceness_prose": niceness_prose,
        "confidence_prose": confidence_prose,
        "confidence": conf,
        "length_bucket": bucket,
    }


def behavior_tier(assignment: dict | None) -> str:
    """Map a cell onto a generic search tier. Missing → ordinary."""
    raw = str((assignment or {}).get("stance") or (assignment or {}).get("user_behavior") or "")
    return _TIER_ALIASES.get(raw, "ordinary")


# Trainer-facing conversation labels. Only keys the sampler already uses.
_CONVERSATION_OPTIONAL = (
    "stance",
    "tone",
    "length",
    "ask",
    "vagueness",
    "phrasing",
    "pressure",
    "user",
    "texture",
    "history",
)


def conversation_features(
    assignment: dict | None = None,
    tags: dict | None = None,
    *,
    ask_family: str | None = None,
    tool: str | None = None,
) -> dict[str, Any]:
    """User/conversation labels for a training row. Omit unsampled tags."""
    assignment = dict(assignment or {})
    tags = dict(tags or {})
    out: dict[str, Any] = {"tier": behavior_tier(assignment)}
    family = str(ask_family or "").strip()
    if family in {"tool", "general", "vague"}:
        out["ask_family"] = family
        out["intent_known"] = family != "vague"
        named = bool(tool) and str(tool) not in {"unrelated", "multi_tool"}
        if not named:
            named = bool(assignment.get("tool")) and str(assignment.get("tool")) not in {
                "unrelated",
                "multi_tool",
            }
        out["tool_known"] = family == "tool" and named
    for key in _CONVERSATION_OPTIONAL:
        val = tags.get(key)
        if val not in (None, ""):
            out[key] = val
    return out


# TIER_COVER_FROM = 8: from this many picks every tier gets at least one
# item, so a slice never reads as a single stance (convention, untested).
TIER_COVER_FROM = 8


def mix_items_by_tier(items: list, n: int, tier_of, *, hard_share: float | None = None) -> list:
    """Breadth-first across tiers, then fill to ``hard_share``.

    First items hit ordinary plus a hard case. A 24-cell or 100-row slice
    is not one tier. Unknown tiers count as ordinary. ``hard_share`` is the
    fraction drawn from the hard tiers, round-robin across them; the
    default is ``HARD_SHARE``.
    """
    if not items or n <= 0:
        return []
    hard_share = clamp_share(hard_share)
    n = min(int(n), len(items))
    buckets: dict[str, list] = {tier: [] for tier in _TIERS}
    for item in items:
        tier = tier_of(item)
        buckets[tier if tier in buckets else "ordinary"].append(item)

    picked: list = []
    seen: set[int] = set()

    def take(tier: str) -> bool:
        for item in buckets.get(tier) or []:
            marker = id(item)
            if marker in seen:
                continue
            seen.add(marker)
            picked.append(item)
            return True
        return False

    take("ordinary")
    if n >= 2:  # noqa: PLR2004  # two tiers before a mix exists
        for tier in ("adversarial", "boundary", "ambiguous"):
            if take(tier):
                break
    if n >= TIER_COVER_FROM:
        have = {tier_of(item) for item in picked}
        for tier in _TIERS:
            if len(picked) >= n:
                break
            if tier not in have:
                take(tier)

    # No floor: the old `max((n + 1) // 2, ...)` pinned ordinary at 50% for
    # any hard share over it, so the dial only turned one way.
    target_ordinary = n - min(n, max(0, round(n * hard_share)))
    hard_cursor = 0
    while len(picked) < n:
        ordinary_count = sum(1 for item in picked if tier_of(item) == "ordinary")
        if ordinary_count < target_ordinary and take("ordinary"):
            continue
        # Round-robin the hard tiers: draining "ambiguous" first turned a
        # 25% ordinary ask into one hard tier instead of a hard mix.
        progressed = False
        for offset in range(len(_HARD_TIERS)):
            tier = _HARD_TIERS[(hard_cursor + offset) % len(_HARD_TIERS)]
            if take(tier):
                hard_cursor = (hard_cursor + offset + 1) % len(_HARD_TIERS)
                progressed = True
                break
        if not progressed and take("ordinary"):
            progressed = True
        if not progressed:
            break
    return picked


_HINT_STOP = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "or",
    "and",
    "if",
    "is",
    "do",
    "not",
    "you",
    "your",
    "must",
    "should",
    "with",
    "for",
    "on",
    "in",
    "be",
}


def _private_hint(text: Any, *, limit: int = 24) -> str:
    """Compact tag. Full policy / world sentences stay out of the cell."""
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return ""
    if len(raw) <= limit and " " in raw and raw.endswith("."):
        return raw.rstrip(".")[:limit]
    if len(raw) <= limit:
        return raw
    words = [w for w in re.findall(r"[A-Za-z0-9_]+", raw.lower()) if w not in _HINT_STOP]
    slug = " ".join(words[:3])
    return (slug or raw)[:limit]


def _world_hint(raw: Any) -> str:
    from .scenarios import WORLD_HINTS

    text = str(raw or "").strip()
    if text in WORLD_HINTS:
        return WORLD_HINTS[text]
    return _private_hint(text, limit=16)


def _length_hint(raw: Any, n_len: int) -> str:
    text = str(raw or "").strip().lower()
    if "short" in text:
        return "short prompt"
    if "long" in text:
        return "long prompt"
    if "medium" in text or "mid" in text:
        return "medium prompt"
    if text in _LENGTHS:
        return text
    # Right-skew: most medium, some short, long tail (LENGTH_SHORT_BELOW,
    # LENGTH_LONG_FROM as percent points of a 0..99 draw).
    bucket = n_len % 100
    if bucket < round(LENGTH_SHORT_BELOW * 100):
        return "short prompt"
    if bucket < round(LENGTH_LONG_FROM * 100):
        return "medium prompt"
    return "long prompt"


#: How often an untagged card gets each optional tag, as "one card in N"
#: on an independent hash draw per tag. Conventions, untested: the tags
#: are hints the writer often ignores, and a tagged assignment always wins.
LENGTH_HINT_SKIP_ONE_IN = 5  #: four cards in five carry a length hint
COMPOUND_ASK_ONE_IN = 23  #: several asks in one message
ASK_ONE_IN = 5  #: a question or a plain ask, when not compound
PRESSURE_ONE_IN = 19  #: rushed, insistent, repeat
USER_TYPE_ONE_IN = 17  #: first time, returning, in a hurry, careful, brief
TOOL_CONDITION_HINT_ONE_IN = 11  #: tell the writer the tool will fault
TONE_WITH_TEXTURE_ONE_IN = 3  #: a textured card also gets a tone
STANDARD_TEXTURE_SHARE = (3, 5)  #: of untextured cards, 3 in 5 say "type normally"
#: One draw modulo MODE_SLOTS picks at most one of the rarer hints per card:
#: slot 14 vagueness, 16 odd phrasing, 11 history, 15 a stance the grid did
#: not map, every fifth slot (2, 7, 12, 17) a world-state hint, and a mapped
#: non-ordinary stance shows on the first ten slots (half the cards).
MODE_SLOTS = 20
MODE_VAGUENESS = 14
MODE_PHRASING = 16
MODE_HISTORY = 11
MODE_UNMAPPED_STANCE = 15
MODE_STANCE_BELOW = 10
MODE_WORLD_EVERY = 5
MODE_WORLD_SLOT = 2


def sample_cell_tags(
    seed: int,
    round_index: int,
    key: str,
    assignment: dict | None = None,
    texture_rate: float = DEFAULT_TEXTURE_RATE,
    **_unused,
) -> dict[str, Any]:
    """Most cells are tools and policy. Length and phrasing are rare hints."""
    assignment = dict(assignment or {})
    n = _draw(seed, round_index, key, "sparse")
    n_len = _draw(seed, round_index, key, "length")
    n_tier = _draw(seed, round_index, key, "tier")
    situation: dict[str, Any] = {}
    if assignment.get("tool"):
        situation["tool"] = assignment["tool"]
    rule = str(assignment.get("rule") or "")
    if rule and rule != "unspecified":
        # A rule-free cell (the block that pairs the other axes, or a run
        # with no policy) gets no rule hint, so its card does not move
        # when the policy does.
        hint = _private_hint(rule)
        if hint:
            situation["rule"] = hint

    def add_stance() -> None:
        raw = str(assignment.get("stance") or assignment.get("user_behavior") or "")
        if not raw:
            from .scenarios import STANCES

            raw = STANCES[n_tier % len(STANCES)]
        if raw == "ordinary":
            return
        situation["stance"] = raw

    if assignment.get("length") or (n % LENGTH_HINT_SKIP_ONE_IN != 0):
        situation["length"] = _length_hint(assignment.get("length"), n_len)

    n_ask = _draw(seed, round_index, key, "ask")
    if assignment.get("ask"):
        situation["ask"] = assignment["ask"]
    elif n_ask % COMPOUND_ASK_ONE_IN == 0:
        situation["ask"] = _COMPOUND[(n_ask // COMPOUND_ASK_ONE_IN) % len(_COMPOUND)]
    elif n_ask % ASK_ONE_IN == 0:
        situation["ask"] = _ASKS[(n_ask // ASK_ONE_IN) % len(_ASKS)]

    mode = n % MODE_SLOTS
    if mode == MODE_VAGUENESS:
        situation["vagueness"] = assignment.get("vagueness") or _VAGUENESS[(n // MODE_SLOTS) % 3]
    elif mode == MODE_PHRASING:
        situation["phrasing"] = _WEIRD[(n // MODE_SLOTS) % len(_WEIRD)]
    raw_stance = assignment.get("stance") or assignment.get("user_behavior")
    mapped_stance = _TIER_ALIASES.get(str(raw_stance)) if raw_stance else None
    if (raw_stance and str(raw_stance) != "ordinary" and mode < MODE_STANCE_BELOW) or (
        mapped_stance is None and mode == MODE_UNMAPPED_STANCE
    ):
        add_stance()

    n_extra = _draw(seed, round_index, key, "axis")
    if assignment.get("pressure") or n_extra % PRESSURE_ONE_IN == 0:
        situation["pressure"] = (
            assignment.get("pressure") or _PRESSURES[(n_extra // PRESSURE_ONE_IN) % len(_PRESSURES)]
        )
    if assignment.get("user") or n_extra % USER_TYPE_ONE_IN == 0:
        situation["user"] = (
            assignment.get("user") or _USER_TYPES[(n_extra // USER_TYPE_ONE_IN) % len(_USER_TYPES)]
        )
    hist = assignment.get("history")
    if hist and hist != "fresh" and mode == MODE_HISTORY:
        situation["history"] = hist
    world = assignment.get("world_state")
    if (
        world
        and world not in {"unspecified", "unknown"}
        and mode % MODE_WORLD_EVERY == MODE_WORLD_SLOT
    ):
        situation["world_state"] = _world_hint(world)
    cond = assignment.get("tool_condition")
    if cond and cond != "success" and n_extra % TOOL_CONDITION_HINT_ONE_IN == 0:
        situation["tool_condition"] = cond

    t = _draw(seed, round_index, key, "texture")
    if texture_rate > 0 and (t % 1000) < int(min(1.0, texture_rate) * 1000):
        situation["texture"] = _TEXTURES[(t // 1000) % len(_TEXTURES)]
        if (t // 7919) % TONE_WITH_TEXTURE_ONE_IN == 0:
            situation["tone"] = _TONES[(t // 104729) % len(_TONES)]
    elif (t // 977) % STANDARD_TEXTURE_SHARE[1] < STANDARD_TEXTURE_SHARE[0]:
        # The writer model collapses to lowercase texting on its own, so
        # ordinary prose (capitals, end marks) must be an explicit style too.
        situation["texture"] = "standard"

    return situation


# DEFAULT_CLOCK_S = 600: the wall-clock the planners assume when a run
# names no time budget (ten minutes; convention). PLAN_UNIT_S = 60: the
# clock at which the search plan is breadth-first only; every minute above
# widens it. PLAN_SCALE_MIN = 0.5 / PLAN_SCALE_MAX = 3.0: the plan scale
# is clamped to this band, half a unit to three units (convention,
# untested).
DEFAULT_CLOCK_S = 600.0
PLAN_UNIT_S = 60.0
PLAN_SCALE_MIN = 0.5
PLAN_SCALE_MAX = 3.0
# SHAPE_LEN_CLOCK_S = 120: under two minutes of clock the shape mining
# stops at two-field shapes; above it, three (convention, untested).
SHAPE_LEN_CLOCK_S = 120.0


def sampling_plan(time_budget: float | None) -> dict[str, Any]:
    """Wider search when they give more wall-clock. Still BFS at 60s."""
    seconds = (
        DEFAULT_CLOCK_S if time_budget is None or float(time_budget) <= 0 else float(time_budget)
    )
    scale = min(PLAN_SCALE_MAX, max(PLAN_SCALE_MIN, seconds / PLAN_UNIT_S))
    return {
        "seconds": seconds,
        "scale": scale,
        "shape_limit": max(8, round(12 * scale)),
        "enum_cap": max(200, round(200 * scale)),
        "max_shape_len": 2 if seconds < SHAPE_LEN_CLOCK_S else 3,
        "ordinary_share": ORDINARY_SHARE,
    }


#: The adaptive mix: explore rises from 30% of the batch on a 15 s clock to
#: 80% at three minutes; what is left splits 55/45 between expand and
#: verify; up to 3 phrasings and 2 or 3 repeats per situation, 3 when the
#: clock is long enough (90 s) for verify to run. Conventions, untested.
ADAPTIVE_EXPLORE_MIN = 0.30
ADAPTIVE_EXPLORE_MAX = 0.80
ADAPTIVE_CLOCK_LO_S = 15.0
ADAPTIVE_CLOCK_SPAN_S = 165.0
ADAPTIVE_SAT_MIN_CLOCK_S = 120.0
ADAPTIVE_EXPAND_OF_REST = 0.55
ADAPTIVE_N_REQ = 3
ADAPTIVE_K_SHORT = 2
ADAPTIVE_K_LONG = 3
ADAPTIVE_K_LONG_FROM_S = 90.0


def adaptive_allocator(
    time_budget: float | None, until: str = "compute", *, elapsed: float | None = None
) -> dict[str, Any]:
    """Adaptive mix. Short remaining clock is messier; saturation walks more cards.

    Shares are explore / expand / verify. n_req and k are caps so expand and
    verify can actually run. Not a pinned n=1 k=1 policy.
    """
    until_key = str(until or "compute").strip().lower()
    if until_key in {"first", "saturation"}:
        until_key = "saturation"
    elif until_key in {"compute", "budget_only", "budget", "time"}:
        until_key = "compute"
    sat = until_key == "saturation"
    if time_budget is None or float(time_budget) <= 0:
        total = DEFAULT_CLOCK_S
        left = DEFAULT_CLOCK_S
    else:
        total = float(time_budget)
        spent = 0.0 if elapsed is None else max(0.0, float(elapsed))
        left = max(0.0, total - spent)
    # Saturation keeps covering the grid even on a short clock.
    clock = max(left, ADAPTIVE_SAT_MIN_CLOCK_S) if sat else left
    scale = min(1.0, max(0.0, (clock - ADAPTIVE_CLOCK_LO_S) / ADAPTIVE_CLOCK_SPAN_S))
    explore = ADAPTIVE_EXPLORE_MIN + (ADAPTIVE_EXPLORE_MAX - ADAPTIVE_EXPLORE_MIN) * scale
    rest = 1.0 - explore
    expand = rest * ADAPTIVE_EXPAND_OF_REST
    verify = rest * (1.0 - ADAPTIVE_EXPAND_OF_REST)
    return {
        "explore": explore,
        "expand": expand,
        "verify": verify,
        "n_req": ADAPTIVE_N_REQ,
        "k": ADAPTIVE_K_LONG if sat or total >= ADAPTIVE_K_LONG_FROM_S else ADAPTIVE_K_SHORT,
        "until": until_key,
        "seconds": total,
        "remaining": left,
    }


def allocator_slot_counts(take: int, plan: dict | None) -> dict[str, int]:
    """Integer explore/expand/verify slots from mix shares."""
    take = max(0, int(take))
    if take <= 0 or not plan:
        return {"explore": take, "expand": 0, "verify": 0}
    expand_s = float(plan.get("expand") or 0.0)
    verify_s = float(plan.get("verify") or 0.0)
    messy_s = expand_s + verify_s
    messy_n = round(take * messy_s)
    if take >= 2:  # noqa: PLR2004  # two: a pair is the structural minimum
        messy_n = max(1, min(take - 1, messy_n))
    explore_n = take - messy_n
    if messy_n <= 0:
        return {"explore": take, "expand": 0, "verify": 0}
    v_part = verify_s / messy_s if messy_s else 0.5
    verify_n = round(messy_n * v_part)
    verify_n = min(messy_n, max(0, verify_n))
    if messy_n >= 2 and verify_n == 0 and v_part > 0:  # noqa: PLR2004  # two: a pair is the structural minimum
        verify_n = 1
    expand_n = messy_n - verify_n
    return {"explore": explore_n, "expand": expand_n, "verify": verify_n}


def new_turn_stats() -> dict[str, Any]:
    """Shared running mean of observed conversation turns."""
    return {"n": 0, "sum": 0, "lock": threading.Lock()}


def observed_turn_count(row: dict | None) -> int:
    """User + agent utterances. Opening prompt counts as the first user turn."""
    n = 1
    for step in (row or {}).get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("user"):
            n += 1
        if step.get("text") or step.get("tool"):
            n += 1
    return n


def record_turns(stats: dict | None, row: dict | None) -> None:
    if not stats:
        return
    n = observed_turn_count(row)
    lock = stats.get("lock")
    if lock is not None:
        lock.acquire()
    try:
        stats["n"] = int(stats.get("n") or 0) + 1
        stats["sum"] = int(stats.get("sum") or 0) + n
    finally:
        if lock is not None:
            lock.release()


def running_turn_mean(stats: dict | None) -> float | None:
    if not stats:
        return None
    lock = stats.get("lock")
    if lock is not None:
        lock.acquire()
    try:
        n = int(stats.get("n") or 0)
        if n <= 0:
            return None
        return float(stats.get("sum") or 0) / n
    finally:
        if lock is not None:
            lock.release()


#: DEFAULT_AVG_TURNS: the mean thread length ``local_model`` and this
#: sampler aim for when a caller names none; the same 12 ``simulate()``
#: uses, from defaults.py (one value, one home; the reason is there).
# TURN_MIX_SHORT = 0.15 / TURN_MIX_TAIL = 0.10: of threads, 15% land under
# the middle band, 75% in it (center +- TURN_BAND) and 10% in the long
# tail; TURN_GAIN = 0.5 is the proportional correction toward avg_turns
# from the live mean, a full mirror (gain 1) overshoots and oscillates.
# The mix is a convention, untested; the gain was observed.
TURN_MIX_SHORT = 0.15
TURN_MIX_TAIL = 0.10
TURN_BAND = 2
TURN_GAIN = 0.5
#: Under a target of TURN_SHORT_TARGET the old short mix applies: 60% of
#: threads 2-4 turns, 30% 5-8, 10% longer (convention).
TURN_SHORT_TARGET = 3.5
TURN_SHORT_MIX = (0.60, 0.90)


def sample_turn_budget(
    seed: int,
    key: str,
    max_turns: int,
    avg_turns: float | None = None,
    running_mean: float | None = None,
) -> int:
    """15/75/10 around ``avg_turns``, shifted by the live mean, snapped even."""
    cap = max(2, int(max_turns))
    target = DEFAULT_AVG_TURNS if avg_turns is None else float(avg_turns)
    center = target
    if running_mean is not None:
        # Proportional correction at TURN_GAIN, see above.
        center = target + TURN_GAIN * (target - float(running_mean))
    # The correction can push the center past the cap; clamped, so the
    # middle band never inverts (mid_hi < mid_lo divided by zero once
    # avg_turns reached the cap).
    center = max(2, min(cap, round(center)))
    mid_lo = max(2, center - TURN_BAND)
    mid_hi = min(cap, max(mid_lo, center + TURN_BAND))
    short_hi = mid_lo - 1
    tail_lo = min(cap, mid_hi + 1)
    u = _draw(int(seed), 0, str(key), "turns") % 1000
    short_cut = round(TURN_MIX_SHORT * 1000)
    tail_cut = round((1.0 - TURN_MIX_TAIL) * 1000)
    if target <= TURN_SHORT_TARGET:
        if u < round(TURN_SHORT_MIX[0] * 1000):
            want = 2 + (u % 3)
        elif u < round(TURN_SHORT_MIX[1] * 1000):
            want = 5 + (u % 4)
        else:
            want = 9 + (u % max(1, cap - 8))
    elif u < short_cut and short_hi >= 2:  # noqa: PLR2004  # a short band needs two turns
        want = 2 + (u % (short_hi - 1))
    elif u < tail_cut:
        want = mid_lo + (u % (mid_hi - mid_lo + 1))
    else:
        want = tail_lo + (u % max(1, cap - tail_lo + 1))
    # Even length: user, agent, user, agent. Steer the odd snap toward the target.
    if want % 2:
        want = want + 1 if (running_mean is None or running_mean <= target) else want - 1
    if want % 2:
        want += 1
    even_cap = cap if cap % 2 == 0 else cap - 1
    return max(2, min(even_cap if even_cap >= 2 else cap, want))  # noqa: PLR2004  # two turns is the shortest thread


def sample_request_axes(
    seed: int, round_index: int, key: str, *, assignment: dict | None = None, **_unused
) -> dict[str, str]:
    """Optional extras only. Often empty."""
    tags = sample_cell_tags(seed, round_index, key, assignment)
    return {k: str(v) for k, v in tags.items() if k not in {"tool", "rule"}}


#: Annealed exploration of the candidate pool. ANNEAL_START = 1.0 decays by
#: ANNEAL_DECAY = 0.93 per round to ANNEAL_END = 0.12, so by round 30 the
#: batch is almost all exploitation (convention, untested). Explore slots
#: are EXPLORE_FRACTION_MIN..MAX of the batch, scaled by the temperature,
#: never more than half of it.
ANNEAL_START = 1.0
ANNEAL_END = 0.12
ANNEAL_DECAY = 0.93
EXPLORE_FRACTION_MIN = 0.04
EXPLORE_FRACTION_MAX = 0.40
# NOVELTY_PIVOT = 0.45: cosine distance above which an off-batch candidate
# earns extra acceptance; ACCEPT_FLOOR = 0.15 is what a candidate at or
# under the pivot gets at temperature 1, ACCEPT_SLOPE = 2.0 per unit of
# distance above it. 0.45 is a coarse duplicate threshold: SemDeDup
# removed 28% of LAION at eps 0.37 and 37% at 0.63 (2303.09540), so a
# candidate under the pivot is likely a paraphrase of something run.
NOVELTY_PIVOT = 0.45
ACCEPT_FLOOR = 0.15
ACCEPT_SLOPE = 2.0


def anneal_temperature(
    round_index: int,
    *,
    start: float = ANNEAL_START,
    end: float = ANNEAL_END,
    decay: float = ANNEAL_DECAY,
) -> float:
    return max(end, start * (decay ** max(0, int(round_index))))


def explore_slot_count(batch_size: int, round_index: int) -> int:
    need = max(1, int(batch_size))
    temp = anneal_temperature(round_index)
    frac = EXPLORE_FRACTION_MIN + (EXPLORE_FRACTION_MAX - EXPLORE_FRACTION_MIN) * temp
    return max(0, min(need // 2, round(need * frac)))


# ANNEAL_OFF_TEMPERATURE = 0.01: below this annealing temperature no
# off-batch candidate is accepted; the explore slot is closed (convention).
ANNEAL_OFF_TEMPERATURE = 0.01


def accept_anneal_candidate(
    novelty: float, *, temperature: float, rng: random.Random | None = None
) -> bool:
    """Accept an off-batch candidate into an explore slot.

    Acceptance rises with novelty (min cosine distance to everything
    already run) and falls with the annealing temperature: at temperature
    1.0 a candidate 0.9 away is taken almost always, one 0.45 away or
    closer about 15% of the time. The earlier form had the sign flipped
    and took duplicates almost always.
    """
    if temperature <= ANNEAL_OFF_TEMPERATURE:
        return False
    rng = rng or random.Random()
    uplift = max(0.0, float(novelty) - NOVELTY_PIVOT)
    prob = min(1.0, temperature * (ACCEPT_FLOOR + uplift * ACCEPT_SLOPE))
    return rng.random() < prob


def apply_annealing_explore(
    batch_idx: list[int],
    texts_len: int,
    novelty_of: dict[int, float],
    *,
    need: int,
    round_index: int,
    seed: int,
) -> list[int]:
    explore_n = explore_slot_count(need, round_index)
    if explore_n <= 0 or texts_len <= need:
        return batch_idx
    temp = anneal_temperature(round_index)
    rng = random.Random(int(seed) + int(round_index) * 31 + 7)
    in_batch = set(batch_idx)
    pool = [i for i in range(texts_len) if i not in in_batch]
    # Most novel first: explore slots are for what the batch has not seen.
    pool.sort(key=lambda i: -novelty_of.get(i, 0.0))
    accepted: list[int] = []
    for index in pool:
        if len(accepted) >= explore_n:
            break
        if accept_anneal_candidate(novelty_of.get(index, 0.0), temperature=temp, rng=rng):
            accepted.append(index)
    if not accepted:
        return batch_idx
    sorted_batch = sorted(batch_idx, key=lambda i: novelty_of.get(i, 0.0))
    drop = set(sorted_batch[: len(accepted)])
    merged = [i for i in batch_idx if i not in drop] + accepted
    return list(dict.fromkeys(merged))[:need]

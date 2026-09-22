"""Quant-style coding tasks over one seeded price table, built offline.

    python tasks.py            # build, split, write out/tasks.jsonl, print the counts

The table is eight tickers over 500 trading days, drawn from
``random.Random(SEED)`` so every machine builds the same bars with nothing
but the standard library (the dry run and CI have no numpy or pandas, and
the grader's sandbox is the same interpreter). Every task is one function to
write over that table: max drawdown, rolling Sharpe, RSI, realized
volatility, a momentum factor's Spearman rank correlation with next-day
returns, a top-k long-short return, golden crosses, beta, ATR, VWAP, the
longest up streak. Parameters vary by template, so there are
``len(build())`` distinct tasks. Each task carries ``privileged.tests``:
asserts against the reference implementation's answer on the seeded table,
computed here at build time. The answer is never in the prompt.

The grader is ``wai.verify.CodeExec(setup=SETUP)``: it pulls the last
```python fence from the reply, prepends ``SETUP`` (the table builder and
``BARS``), appends the tests, and runs the file in a fresh interpreter.
Reward 1 when it exits 0.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

SEED = 0  # the table's draw and the split's draw
N_DAYS = 500  # trading days in the table
TICKERS = ("ALP", "BRV", "CHR", "DLT", "ECH", "FOX", "GLF", "HTL")
HOLDOUT = (
    0.5  # share of tasks held out, split by task id (the train split picks, the holdout decides)
)
TRADING_DAYS = 252  # annualisation factor for Sharpe and realized volatility
TOLERANCE = 1e-6  # a float answer passes within this of the reference

# The table builder as source, so the same text defines BARS at build time
# here and in the grader's sandbox (CodeExec prepends SETUP to every reply).
SETUP = f'''
import math, random

def make_bars(seed={SEED}, n_days={N_DAYS}, tickers={TICKERS!r}):
    """Eight tickers, {N_DAYS} trading days, one dict per (date, ticker), sorted by date then ticker."""
    rng = random.Random(seed)
    import datetime
    day = datetime.date(2020, 1, 1)
    dates = []
    while len(dates) < n_days:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += datetime.timedelta(days=1)
    bars = []
    for ticker in tickers:
        price = rng.uniform(20.0, 200.0)
        drift = rng.uniform(-0.0005, 0.0010)
        vol = rng.uniform(0.008, 0.030)
        base_volume = rng.uniform(2e5, 5e6)
        for date in dates:
            open_ = price * (1.0 + rng.gauss(0.0, vol / 4))
            close = open_ * math.exp(drift + rng.gauss(0.0, vol))
            wick_up = abs(rng.gauss(0.0, vol / 2))
            wick_down = abs(rng.gauss(0.0, vol / 2))
            high = max(open_, close) * (1.0 + wick_up)
            low = min(open_, close) * (1.0 - wick_down)
            volume = int(base_volume * math.exp(rng.gauss(0.0, 0.4)))
            bars.append(
                {{
                    "date": date,
                    "ticker": ticker,
                    "open": round(open_, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": volume,
                }}
            )
            price = close
    bars.sort(key=lambda b: (b["date"], b["ticker"]))
    return bars

BARS = make_bars()
'''

_ns: dict[str, Any] = {}
exec(SETUP, _ns)  # the same source the sandbox runs
BARS: list[dict[str, Any]] = _ns["BARS"]

DATA_SHAPE = (
    "`bars` is a list of dicts, one per (date, ticker), sorted by date then ticker, with keys "
    "`date` (an ISO string, 'YYYY-MM-DD'), `ticker` (str), `open`, `high`, `low`, `close` "
    "(floats) and `volume` (int). Eight tickers, 500 trading days, no gaps. "
)
REPLY_SHAPE = (
    "Reply with one ```python block that defines the function. Standard library only; the "
    "table is passed in as `bars`, so do not read files or generate data."
)


# --------------------------------------------------------------------------
# Reference implementations. Plain Python so the build needs nothing.
# --------------------------------------------------------------------------


def _closes(bars: list[dict], ticker: str, field: str = "close") -> list[float]:
    return [float(b[field]) for b in bars if b["ticker"] == ticker]


def _returns(closes: list[float]) -> list[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    for rank, i in enumerate(order, start=1):
        ranks[i] = float(rank)
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy)


def ref_max_drawdown(bars: list[dict], ticker: str) -> float:
    closes = _closes(bars, ticker)
    peak, worst = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        worst = min(worst, c / peak - 1.0)
    return worst


def ref_rolling_sharpe(bars: list[dict], ticker: str, window: int) -> float:
    r = _returns(_closes(bars, ticker))[-window:]
    return _mean(r) / _std(r) * math.sqrt(TRADING_DAYS)


def ref_rsi(bars: list[dict], ticker: str, period: int) -> float:
    closes = _closes(bars, ticker)
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))][-period:]
    gain = _mean([max(c, 0.0) for c in changes])
    loss = _mean([max(-c, 0.0) for c in changes])
    if loss == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def ref_realized_vol(bars: list[dict], ticker: str, window: int) -> float:
    closes = _closes(bars, ticker)
    logs = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))][-window:]
    return _std(logs) * math.sqrt(TRADING_DAYS)


def _panel(bars: list[dict]) -> tuple[list[str], dict[str, list[float]]]:
    tickers = sorted({b["ticker"] for b in bars})
    return tickers, {t: _closes(bars, t) for t in tickers}


def _cross_section(bars: list[dict], lookback: int, skip: int):
    """Per date: (momentum by ticker, next-day return by ticker)."""
    tickers, closes = _panel(bars)
    n = len(closes[tickers[0]])
    for t in range(lookback, n - 1):
        mom = [closes[k][t - skip] / closes[k][t - lookback] - 1.0 for k in tickers]
        nxt = [closes[k][t + 1] / closes[k][t] - 1.0 for k in tickers]
        yield mom, nxt


def ref_momentum_ic(bars: list[dict], lookback: int, skip: int) -> float:
    ics = [_pearson(_ranks(m), _ranks(n)) for m, n in _cross_section(bars, lookback, skip)]
    return _mean(ics)


def ref_long_short_return(bars: list[dict], lookback: int, k: int) -> float:
    out = []
    for mom, nxt in _cross_section(bars, lookback, 0):
        order = sorted(range(len(mom)), key=lambda i: mom[i])
        out.append(_mean([nxt[i] for i in order[-k:]]) - _mean([nxt[i] for i in order[:k]]))
    return _mean(out)


def _sma(xs: list[float], window: int) -> list[float]:
    return [_mean(xs[i - window + 1 : i + 1]) for i in range(window - 1, len(xs))]


def ref_golden_crosses(bars: list[dict], ticker: str, fast: int, slow: int) -> int:
    closes = _closes(bars, ticker)
    f = _sma(closes, fast)[slow - fast :]
    s = _sma(closes, slow)
    return sum(1 for i in range(1, len(s)) if f[i] > s[i] and f[i - 1] <= s[i - 1])


def ref_beta(bars: list[dict], ticker: str, market: str) -> float:
    r = _returns(_closes(bars, ticker))
    m = _returns(_closes(bars, market))
    mr, mm = _mean(r), _mean(m)
    cov = sum((a - mr) * (b - mm) for a, b in zip(r, m))
    var = sum((b - mm) ** 2 for b in m)
    return cov / var


def ref_atr(bars: list[dict], ticker: str, period: int) -> float:
    rows = [b for b in bars if b["ticker"] == ticker]
    trs = []
    for i in range(1, len(rows)):
        prev = rows[i - 1]["close"]
        h, lo = rows[i]["high"], rows[i]["low"]
        trs.append(max(h - lo, abs(h - prev), abs(lo - prev)))
    return _mean(trs[-period:])


def ref_vwap(bars: list[dict], ticker: str, window: int) -> float:
    rows = [b for b in bars if b["ticker"] == ticker][-window:]
    return sum(b["close"] * b["volume"] for b in rows) / sum(b["volume"] for b in rows)


def ref_max_up_streak(bars: list[dict], ticker: str) -> int:
    closes = _closes(bars, ticker)
    best = run = 0
    for i in range(1, len(closes)):
        run = run + 1 if closes[i] > closes[i - 1] else 0
        best = max(best, run)
    return best


# --------------------------------------------------------------------------
# Templates: (family, function name, parameter names, the spec, reference,
# the parameter grid). The spec is what the prompt says; it defines the
# quantity exactly, never its value.
# --------------------------------------------------------------------------

_T = tuple[str, str, tuple[str, ...], str, Callable[..., Any], list[tuple[Any, ...]]]

TEMPLATES: list[_T] = [
    (
        "max_drawdown",
        "max_drawdown",
        ("ticker",),
        "Return the maximum drawdown of ticker `{ticker}` as a negative fraction: the minimum "
        "over days t of close_t / (the highest close from the first day through day t) - 1, "
        "using only that ticker's rows in date order. A series that never falls returns 0.0.",
        ref_max_drawdown,
        [(t,) for t in TICKERS],
    ),
    (
        "rolling_sharpe",
        "rolling_sharpe",
        ("ticker", "window"),
        "Return the trailing Sharpe ratio of ticker `{ticker}` over its last {window} simple "
        "daily returns (r_t = close_t / close_(t-1) - 1): the mean of those returns divided "
        "by their sample standard deviation (n - 1 in the denominator), times sqrt(252).",
        ref_rolling_sharpe,
        [(t, w) for t in TICKERS for w in (20, 60)],
    ),
    (
        "rsi",
        "rsi",
        ("ticker", "period"),
        "Return the RSI of ticker `{ticker}` over its last {period} daily close changes "
        "(d_t = close_t - close_(t-1)): average gain is the mean of the changes with losses "
        "counted as 0, average loss is the mean of the absolute losses with gains counted as "
        "0, RSI = 100 - 100 / (1 + average gain / average loss); return 100.0 when the "
        "average loss is 0. Simple averages, not Wilder smoothing.",
        ref_rsi,
        [(t, p) for t in TICKERS for p in (14, 28)],
    ),
    (
        "realized_vol",
        "realized_vol",
        ("ticker", "window"),
        "Return the annualised realized volatility of ticker `{ticker}`: the sample standard "
        "deviation (n - 1) of its last {window} daily log returns ln(close_t / close_(t-1)), "
        "times sqrt(252).",
        ref_realized_vol,
        [(t, w) for t in TICKERS for w in (10, 20, 60)],
    ),
    (
        "momentum_ic",
        "momentum_ic",
        ("lookback", "skip"),
        "Return the mean cross-sectional Spearman rank correlation between a momentum factor "
        "and the next-day return. For every day index t (t >= {lookback}, t + 1 exists), for "
        "each ticker: momentum = close_(t-{skip}) / close_(t-{lookback}) - 1 and next-day "
        "return = close_(t+1) / close_t - 1. Rank both across the tickers (1 = smallest, no "
        "ties in this table) and take the Pearson correlation of the two rank vectors; return "
        "the mean over all such days.",
        ref_momentum_ic,
        [(lb, s) for lb in (5, 10, 20, 60) for s in (0, 1)],
    ),
    (
        "long_short_return",
        "long_short_return",
        ("lookback", "k"),
        "Return the mean daily return of an equal-weight top-{k} long, bottom-{k} short "
        "momentum portfolio. For every day index t (t >= {lookback}, t + 1 exists), rank the "
        "tickers by momentum close_t / close_(t-{lookback}) - 1; the day's return is the "
        "mean next-day return (close_(t+1) / close_t - 1) of the {k} highest-momentum tickers "
        "minus that of the {k} lowest; return the mean over days.",
        ref_long_short_return,
        [(lb, k) for lb in (5, 20, 60) for k in (1, 2, 3)],
    ),
    (
        "golden_crosses",
        "golden_crosses",
        ("ticker", "fast", "slow"),
        "Return the number of golden crosses for ticker `{ticker}`: with SMA_f the simple "
        "moving average of close over {fast} days and SMA_s over {slow} days (a day has a "
        "value once {slow} closes exist), count the days t where SMA_f(t) > SMA_s(t) and "
        "SMA_f(t-1) <= SMA_s(t-1). Return an int.",
        ref_golden_crosses,
        [(t, f, s) for t in TICKERS for (f, s) in ((5, 20), (10, 50))],
    ),
    (
        "beta",
        "beta",
        ("ticker", "market"),
        "Return the beta of ticker `{ticker}` to ticker `{market}`: with r and m their simple "
        "daily returns (close_t / close_(t-1) - 1) over every day that has one, beta = "
        "sum((r - mean r) * (m - mean m)) / sum((m - mean m)^2).",
        ref_beta,
        [(t, m) for m in ("ALP", "HTL") for t in TICKERS if t != m],
    ),
    (
        "atr",
        "atr",
        ("ticker", "period"),
        "Return the average true range of ticker `{ticker}` over its last {period} days: the "
        "true range on day t is max(high_t - low_t, |high_t - close_(t-1)|, "
        "|low_t - close_(t-1)|), and the ATR is the simple mean of the last {period} true "
        "ranges.",
        ref_atr,
        [(t, p) for t in TICKERS for p in (14,)],
    ),
    (
        "vwap",
        "vwap",
        ("ticker", "window"),
        "Return the volume-weighted average close of ticker `{ticker}` over its last {window} "
        "days: sum(close * volume) / sum(volume).",
        ref_vwap,
        [(t, w) for t in TICKERS for w in (5, 20)],
    ),
    (
        "max_up_streak",
        "max_up_streak",
        ("ticker",),
        "Return the longest run of consecutive days on which the close of ticker `{ticker}` "
        "was strictly above the previous day's close. Return an int.",
        ref_max_up_streak,
        [(t,) for t in TICKERS],
    ),
]


def _prompt(fn: str, names: tuple[str, ...], spec: str, params: dict[str, Any]) -> str:
    sig = ", ".join(["bars", *names])
    call = ", ".join(["bars", *(repr(params[n]) for n in names)])
    return (
        f"Write a Python function `{fn}({sig})`. {DATA_SHAPE}"
        f"{spec.format(**params)} The tests call `{fn}({call})`. {REPLY_SHAPE}"
    )


def _tests(fn: str, names: tuple[str, ...], params: dict[str, Any], expected: Any) -> str:
    args = ", ".join(["BARS", *(repr(params[n]) for n in names)])
    if isinstance(expected, int):
        return f"out = {fn}({args})\nassert isinstance(out, int)\nassert out == {expected!r}\n"
    return (
        f"out = {fn}({args})\nassert isinstance(out, float)\n"
        f"assert abs(out - ({expected!r})) < {TOLERANCE!r}\n"
    )


def build() -> list[dict[str, Any]]:
    """Every task: ``scenario_id``, ``family``, ``prompt``, ``params``,
    ``privileged.tests`` and ``privileged.reference`` (the reference source
    name, for the offline stand-in). Deterministic: the same list every call."""
    tasks: list[dict[str, Any]] = []
    for family, fn, names, spec, ref, grid in TEMPLATES:
        for i, values in enumerate(grid):
            params = dict(zip(names, values))
            expected = ref(BARS, *values)
            if isinstance(expected, float) and not math.isfinite(expected):
                raise SystemExit(f"{family} {params}: reference is not finite")
            tasks.append(
                {
                    "scenario_id": f"{family}-{i:03d}",
                    "family": family,
                    "function": fn,
                    "params": params,
                    "prompt": _prompt(fn, names, spec, params),
                    "privileged": {
                        "tests": _tests(fn, names, params, expected),
                        "reference": ref.__name__,
                    },
                }
            )
    ids = [t["scenario_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate task ids")
    return tasks


def reference_source(task: dict[str, Any]) -> str:
    """The reference implementation as the reply a perfect model would give:
    a ```python fence defining the task's function. Used only by the offline
    stand-in; the prompt never carries it."""
    import inspect

    ref = globals()[task["privileged"]["reference"]]
    body = inspect.getsource(ref).replace(f"def {ref.__name__}(", f"def {task['function']}(")
    helpers = [
        inspect.getsource(h)
        for h in (
            _closes,
            _returns,
            _mean,
            _std,
            _ranks,
            _pearson,
            _panel,
            _cross_section,
            _sma,
        )
    ]
    return "```python\nimport math\nTRADING_DAYS = 252\n" + "\n".join(helpers) + "\n" + body + "```"


SPLIT = (
    "family"  # "family": whole templates held out; "task": tasks of every template on both sides
)


def split(
    tasks: list[dict[str, Any]], holdout: float = HOLDOUT, seed: int = SEED, by: str = SPLIT
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train and holdout, shuffled by ``seed``, a task wholly on one side.

    ``by="family"`` (the default) holds out whole templates: families are
    drawn in a seeded order until the holdout carries ``holdout`` of the
    tasks. ``by="task"`` shuffles task ids instead. The family split is the
    default because of a measurement, not a preference: two tasks from one
    template differ in a ticker and a window, and ``wai.decontaminate``'s
    8-gram rule (Lambert 2025, chapter Evaluation) reads them as near-copies.
    On this table the task split drops 71 of 71 train rows; the family split
    drops 0 of 64. The family split is also the harder question: the holdout
    asks for quantities the training never showed.
    """
    if by == "task":
        ids = sorted(t["scenario_id"] for t in tasks)
        random.Random(seed).shuffle(ids)
        hold_ids = set(ids[: round(len(ids) * holdout)])
        return [t for t in tasks if t["scenario_id"] not in hold_ids], [
            t for t in tasks if t["scenario_id"] in hold_ids
        ]
    if by != "family":
        raise ValueError("split is 'family' or 'task'")
    families = sorted({t["family"] for t in tasks})
    random.Random(seed).shuffle(families)
    target = round(len(tasks) * holdout)
    hold_fams: set[str] = set()
    count = 0
    for family in families:
        if count >= target:
            break
        hold_fams.add(family)
        count += sum(1 for t in tasks if t["family"] == family)
    return [t for t in tasks if t["family"] not in hold_fams], [
        t for t in tasks if t["family"] in hold_fams
    ]


def write(tasks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")


if __name__ == "__main__":
    all_tasks = build()
    train, hold = split(all_tasks)
    write(all_tasks, HERE / "out" / "tasks.jsonl")
    print(
        f"{len(all_tasks)} tasks in {len(TEMPLATES)} families: {len(train)} train, {len(hold)} holdout"
    )
    print(f"first prompt: {all_tasks[0]['prompt'][:160]}...")
    sys.exit(0)

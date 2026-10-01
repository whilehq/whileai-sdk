"""The `wai` command (`whileai` runs the same entry point): accounts, and the platform objects a coding agent
manages from a terminal.

    wai compare --demo
    wai compare --model ollama:qwen3:4b-instruct --before old.txt --after new.txt         --tasks tasks.jsonl --reward Numeric
    wai login | signup --email | status | logout
    wai init-evals
    wai agents
    wai agent refund-bot
    wai runs refund-bot
    wai verdict refund-bot [--behavior refunds]
    wai promote refund-bot v4
    wai archive refund-bot run_1a2b3c [--undo]
    wai keys
    wai live refund-bot --day 2026-09-17 --version v3 --replies 2400 --flagged 98

``compare`` runs on your machine and needs no account. Platform commands print JSON (``--json``) or a short table, and exit 1 on
an API error with the reason on stderr. They are thin calls into
``whileai.platform``; nothing here talks to anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from . import auth
from .init_evals import add_arguments as init_evals_args


def _platform():
    from . import platform

    return platform


def _fail(err: Exception) -> int:
    print(f"error: {err}", file=sys.stderr)
    return 1


def _emit(payload: Any, as_json: bool, table: Callable[[], None]) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        table()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wai", description="While SDK (the `wai` command; `whileai` is the same)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="sign in from this terminal (opens the browser)")
    p_login.add_argument("--name", help="name for the key on your account (default: cli <host>)")
    p_login.add_argument("--no-browser", action="store_true", help="print the link only")
    p_login.add_argument(
        "--no-wait", action="store_true", help="return after printing the link; run again to finish"
    )
    p_login.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds to wait for approval (default: until the code expires)",
    )

    p_signup = sub.add_parser("signup", help="create an account and a key, no browser")
    p_signup.add_argument("--email", required=True, help="address for the new account")
    p_signup.add_argument("--name", help="name for the key on the account (default: cli <host>)")

    sub.add_parser("logout", help="delete the saved key")
    sub.add_parser("status", help="show which key the SDK will use")

    p_repo = sub.add_parser(
        "init",
        help="set this repo up for a coding agent: AGENTS.md block, CLAUDE.md include, tested skills",
    )
    p_repo.add_argument("--dir", default=".", help="repository root (default: here)")
    p_repo.add_argument(
        "--skill",
        action="append",
        dest="skills",
        help="a skill to install under .claude/skills/ (repeatable; default: the evals playbooks)",
    )
    p_repo.add_argument("--no-check", action="store_true", help="skip running the evals check.py")

    p_init = sub.add_parser(
        "init-evals",
        help="write an eval harness (agent, judge, run, test) wired to this project",
    )
    init_evals_args(p_init)

    p_compare = sub.add_parser(
        "compare",
        help="did my prompt rewrite help: run old and new prompts locally, print the verdict",
    )
    _compare_args(p_compare)

    def platform_parser(name: str, help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--json", action="store_true", help="print the API's JSON")
        p.add_argument("--api-key", help="use this key instead of the saved one")
        return p

    platform_parser("agents", "list the agents tracked on your account")
    p_agent = platform_parser("agent", "one tracked agent: record, behaviors, verdict")
    p_agent.add_argument("id")
    p_runs = platform_parser("runs", "the training runs of one agent, newest first")
    p_runs.add_argument("id")
    p_runs.add_argument("--archived", action="store_true", help="include archived runs")
    p_verdict = platform_parser(
        "verdict", "does the candidate beat the served version, and is it real"
    )
    p_verdict.add_argument("id")
    p_verdict.add_argument(
        "--behavior", help="which behavior's test (default: the latest run's target)"
    )
    p_promote = platform_parser("promote", "make a version the served one")
    p_promote.add_argument("id")
    p_promote.add_argument("version")
    p_archive = platform_parser(
        "archive", "take a run out of the experiment (kept; --undo brings it back)"
    )
    p_archive.add_argument("id")
    p_archive.add_argument("run", help="the run id (wai runs <id> lists them)")
    p_archive.add_argument("--undo", action="store_true", help="unarchive instead")
    platform_parser("keys", "list the API keys on your account (names and prefixes)")
    p_live = platform_parser("live", "report one day of traffic on the served version")
    p_live.add_argument("id")
    p_live.add_argument("--day", required=True, help="YYYY-MM-DD")
    p_live.add_argument("--version", required=True, help="the version that served that day")
    p_live.add_argument("--replies", type=int, required=True)
    p_live.add_argument("--flagged", type=int, default=0, help="replies that failed a check")
    p_live.add_argument("--p50", type=float, default=None, help="median latency in seconds")
    p_live.add_argument("--cost", type=float, default=None, help="USD spent that day")

    args = parser.parse_args(argv)

    if args.command == "login":
        try:
            key = auth.login(
                name=args.name,
                wait=not args.no_wait,
                timeout=args.timeout,
                open_browser=not args.no_browser,
            )
        except auth.LoginError as err:
            return _fail(err)
        return 0 if key or args.no_wait else 2

    if args.command == "signup":
        try:
            auth.signup(args.email, name=args.name)
        except auth.LoginError as err:
            return _fail(err)
        return 0

    if args.command == "init":
        from . import init_repo

        return init_repo.init(
            args.dir,
            skills=args.skills or init_repo.DEFAULT_SKILLS,
            check=not args.no_check,
        )

    if args.command == "init-evals":
        from .init_evals import init_evals

        return init_evals(
            agent=args.agent,
            tools=args.tools,
            system_prompt=args.system_prompt,
            out=args.dir,
            force=args.force,
        )

    if args.command == "compare":
        return _compare(args)

    if args.command == "logout":
        print("Logged out." if auth.logout() else "No saved key.")
        return 0

    if args.command == "status":
        from . import init_repo

        shown = auth.status()
        shown["repo"] = init_repo.status(".")
        print(json.dumps(shown, indent=2))
        if not shown.get("configured"):
            # Not an error: the library runs without an account (CONSTITUTION
            # belief 4). A key buys the hosted writer, judge and run page only,
            # so this goes to stdout as a note, not to stderr as a failure
            # (#790).
            print(
                "no API key: the library runs without one. A key adds the hosted writer "
                "and judge and the run page: `wai login`, or set WHILEAI_API_KEY. "
                "Without one, pass simulator=False and your own agent and judge."
            )
        return 0

    return _platform_command(args)


def _compare_args(p: argparse.ArgumentParser) -> None:
    from .before_after import SEED, TEMPERATURE, K

    p.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in scripted model on 24 arithmetic tasks: offline, no Ollama, seconds",
    )
    p.add_argument(
        "--model",
        help="where the prompts run: ollama:<model>, vllm:<model>@<url>, openai:<model>, ...",
    )
    p.add_argument("--before", help="the old system prompt: a file path, or the text itself")
    p.add_argument("--after", help="the new system prompt: a file path, or the text itself")
    p.add_argument("--tasks", help='JSONL test set, one {"prompt": ..., "reference": ...} per line')
    p.add_argument(
        "--reward",
        default="Numeric",
        help="a wai.verify class (Numeric, ExactMatch, Includes, MathEqual, ...) or "
        "module:function (default: Numeric)",
    )
    p.add_argument("--k", type=int, default=K, help=f"replies per task per arm (default: {K})")
    p.add_argument(
        "--seed", type=int, default=SEED, help=f"seed for sampling and bootstrap (default: {SEED})"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"sampling temperature (default: {TEMPERATURE})",
    )
    p.add_argument("--json", action="store_true", help="print the report as JSON")


def _prompt_text(value: str | None) -> str | None:
    """A file's text when ``value`` names a file, else ``value`` itself."""
    if value is None:
        return None
    from pathlib import Path

    path = Path(value)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:  # a long prompt is not a valid path on every OS
        pass
    return value


def _reward(name: str) -> Any:
    """``Numeric`` -> ``wai.verify.Numeric()``; ``pkg.mod:fn`` -> that callable."""
    import importlib

    if ":" in name:
        module, _, attr = name.partition(":")
        return getattr(importlib.import_module(module), attr)
    from .simulations import verify

    cls = getattr(verify, name, None)
    if cls is None:
        raise ValueError(
            f"--reward {name!r} is not a wai.verify class; use Numeric, ExactMatch, Includes, "
            "MathEqual, MultipleChoice, or module:function"
        )
    return cls()


def _compare(args: argparse.Namespace) -> int:
    from . import before_after

    try:
        if args.demo:
            model: Any = before_after.demo_model
            before: str | None = before_after.DEMO_BEFORE
            after: str | None = before_after.DEMO_AFTER
            tasks: Any = before_after.demo_tasks()
        else:
            missing = [f"--{n}" for n in ("model", "tasks") if not getattr(args, n)]
            if missing:
                raise ValueError(
                    f"{' and '.join(missing)} required (or --demo for the offline example): "
                    "wai compare --model ollama:qwen3:4b-instruct --before old.txt "
                    "--after new.txt --tasks tasks.jsonl"
                )
            model = args.model
            before = _prompt_text(args.before)
            after = _prompt_text(args.after)
            tasks = args.tasks
        report = before_after.compare(
            before,
            after,
            tasks,
            _reward(args.reward),
            model=model,
            k=args.k,
            seed=args.seed,
            temperature=args.temperature,
        )
    except (ValueError, TypeError, OSError, ImportError) as err:
        return _fail(err)
    return _emit(report, args.json, lambda: print(report))


def _platform_command(args: argparse.Namespace) -> int:
    platform = _platform()
    key = args.api_key
    try:
        if args.command == "agents":
            agents = platform.tracked_agents(api_key=key)

            def table():
                if not agents:
                    print("no tracked agents yet: track one with whileai.platform.track(...)")
                for a in agents:
                    print(f"{a.id:24} {a.model or '-':28} serving {a.serving or '-'}")

            return _emit([a.model_dump() for a in agents], args.json, table)

        tracked = platform.Tracked(getattr(args, "id", ""), api_key=key)

        if args.command == "agent":
            dash = tracked.dashboard()
            behaviors = tracked.behaviors()

            def table():
                a = dash.agent
                print(
                    f"{a.id}  model {a.model or '-'}  serving {a.serving or '-'}  candidate {a.candidate or '-'}"
                )
                for b in behaviors:
                    n = f" n={b.n}" if b.n else ""
                    print(f"  {b.name:24} test {b.test_version or '-'}{n}")
                print(str(dash.verdict))

            payload = {
                "agent": dash.agent.model_dump(),
                "behaviors": [b.model_dump() for b in behaviors],
                "verdict": dash.verdict.model_dump(),
            }
            return _emit(payload, args.json, table)

        if args.command == "runs":
            runs = tracked.runs(archived=args.archived)

            def table():
                if not runs:
                    print("no runs yet")
                for r in runs:
                    targets = ",".join(r.get("targets") or []) or "-"
                    print(
                        f"{r.get('version', '-'):10} {r.get('method') or '-':6} {('archived' if r.get('archived') else r.get('status')) or '-':10} targets {targets}  {str(r.get('createdAt', ''))[:10]}"
                    )

            return _emit(runs, args.json, table)

        if args.command == "verdict":
            verdict = tracked.verdict(args.behavior)
            return _emit(verdict.model_dump(), args.json, lambda: print(str(verdict)))

        if args.command == "promote":
            out = tracked.promote(args.version)
            return _emit(out, args.json, lambda: print(f"{args.id}: {args.version} is now serving"))

        if args.command == "archive":
            out = tracked.archive(args.run, archived=not args.undo)
            word = "back in the experiment" if args.undo else "archived"
            return _emit(out, args.json, lambda: print(f"{args.id}: {args.run} {word}"))

        if args.command == "keys":
            out = platform._request("GET", "/keys", api_key=platform._key(key))

            def table():
                for k in out.get("keys") or []:
                    print(
                        f"{k.get('name', '-'):20} {k.get('key', '-'):20} {k.get('tier', '-'):6} {str(k.get('createdAt', ''))[:10]}"
                    )
                print(
                    f"{len(out.get('keys') or [])} of {out.get('limit', 5)}; create or revoke under Account on the platform"
                )

            return _emit(out, args.json, table)

        if args.command == "live":
            out = tracked.live(
                args.day,
                version=args.version,
                replies=args.replies,
                flagged=args.flagged,
                p50_s=args.p50,
                cost_usd=args.cost,
            )
            return _emit(
                out, args.json, lambda: print(f"{args.id}: {args.day} recorded for {args.version}")
            )
    except platform.PlatformError as err:
        return _fail(err)
    except ValueError as err:  # pydantic validation
        return _fail(err)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

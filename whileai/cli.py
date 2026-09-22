"""The `wai` command (`whileai` runs the same entry point): accounts, and the platform objects a coding agent
manages from a terminal.

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

Platform commands print JSON (``--json``) or a short table, and exit 1 on
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

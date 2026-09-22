"""Sign in or sign up from a terminal or a coding agent.

``wai login`` for an existing account (device flow, one click in the
browser). ``wai signup --email`` for a new one: no browser at all, the
account and the key are created in one call.

Device authorization flow (RFC 8628 shape) against the While platform
API. The CLI asks the API for a code pair, prints a link and a short code,
and polls until the human has signed in and pressed Approve in the browser.
The API key that comes back is written to ``~/.whileai/credentials.json``
and every SDK call reads it from there when ``WHILEAI_API_KEY`` is unset.

A pending login survives the process: if the harness running the command
stops it before approval, the next ``wai login`` resumes the same code
instead of printing a new one.

Stdlib only.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path

from whileai._env import env_name, getenv

#: The While platform API (whilehq/website/backend): serves /device/code,
#: /device/token, /signup and /me. ``WHILEAI_API_URL`` overrides.
DEFAULT_API_URL = "https://api.while.ai"
#: Hosts a credentials file may still pin: the token gate this API replaced
#: for login (whileai before 0.5x) and the raw API Gateway hostname the
#: platform API answered on before it had its own name (0.72 to 1.09). Both
#: mean the default now; the key is the same key.
_RETIRED_API_URL = "https://api.zeroproofai.com"
_PREVIOUS_API_URLS = (
    _RETIRED_API_URL,
    "https://mbxp83jd48.execute-api.us-east-1.amazonaws.com",
    "https://api.withwhile.com",
)
#: The site that hosts /sign-in and the /device approval page.
SITE_URL = "https://while.ai"
SIGN_IN_URL = f"{SITE_URL}/sign-in"
#: the trial allowance the gate hands out, used when the reply does not say
DEFAULT_TRIAL_INPUT_TOKENS = 25_000
#: Input tokens one hosted situation spends, measured on a 4-tool spec: a
#: 12-situation run of one cost 28,490, so round it to 2,000 a situation.
#: The trial allowance is small enough that the count is the first thing
#: anyone needs to know about it.
INPUT_TOKENS_PER_SITUATION = 2_000


class LoginError(RuntimeError):
    pass


def trial_situations(daily_input_tokens: float | None = None) -> int:
    """About how many hosted situations a trial day buys, in round numbers."""
    tokens = float(daily_input_tokens or DEFAULT_TRIAL_INPUT_TOKENS)
    return max(1, int(tokens / INPUT_TOKENS_PER_SITUATION))


def trial_note(daily_input_tokens: float | None = None) -> str:
    """The two facts a trial key needs before its first hosted run: how
    far the daily allowance goes, and the offline writer that has no
    allowance at all."""
    return (
        f"That is about {trial_situations(daily_input_tokens)} hosted situations a day "
        f"(a 4-tool spec spends around {INPUT_TOKENS_PER_SITUATION:,} input tokens per "
        "situation); simulate(..., simulator=False) writes situations offline with no quota "
        f"and no network; signing in once at {SIGN_IN_URL} lifts the limit."
    )


def trial_prerun_note() -> str | None:
    """The one line a trial key needs before a hosted run spends it, or ``None``.

    Read from the tier the credentials file recorded at sign-up or at the
    last ``wai status`` / ``wai login``, so a run can say this
    without a network call. A key from the environment has no recorded
    tier, so this says nothing rather than guess at one.
    """
    if getenv("API_KEY"):
        return None
    saved = _read_credentials() or {}
    if not saved.get("api_key") or str(saved.get("tier") or "") != "trial":
        return None
    tokens = int(float(saved.get("daily_input_tokens") or DEFAULT_TRIAL_INPUT_TOKENS))
    return (
        f"trial key: the hosted writer covers about {trial_situations(tokens)} situations a day "
        f"({tokens // 1000}k input tokens); simulator=False writes them offline with no quota; "
        f"sign in once at {SIGN_IN_URL} to lift it"
    )


def _api_url() -> str:
    return getenv("API_URL", DEFAULT_API_URL).rstrip("/")


def config_dir() -> Path:
    """``$WHILEAI_HOME``, else ``~/.whileai``."""
    override = getenv("HOME")
    return Path(override) if override else Path.home() / ".whileai"


def credentials_path() -> Path:
    return config_dir() / "credentials.json"


def _pending_path() -> Path:
    return config_dir() / "pending-login.json"


def _write_private(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with contextlib.suppress(OSError):  # Windows
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_credentials() -> dict | None:
    """The saved credentials file, or ``None``."""
    saved = _read(credentials_path())
    # A login saved against a previous host moves to the default API
    # without signing in again; the key is the same key.
    if saved and saved.get("api_url") in _PREVIOUS_API_URLS:
        saved["api_url"] = DEFAULT_API_URL
    return saved


def _tier_fields(payload: dict) -> dict:
    """The tier facts worth keeping next to the key, from ``/signup`` or ``/me``."""
    tier = str((payload or {}).get("tier") or "").strip()
    if not tier:
        return {}
    fields: dict = {"tier": tier}
    trial = (payload or {}).get("trial") or {}
    if isinstance(trial, dict):
        if trial.get("daily_input_tokens") is not None:
            fields["daily_input_tokens"] = trial["daily_input_tokens"]
        if trial.get("expires_at"):
            fields["expires_at"] = str(trial["expires_at"])
    return fields


def remember_account(payload: dict) -> None:
    """Record a ``/signup`` or ``/me`` reply's tier in the credentials file.

    Only for the key that file holds: a key from the environment may belong
    to another account, and writing its tier here would mislabel this one.
    """
    fields = _tier_fields(payload)
    saved = _read_credentials()
    if not fields or not saved or not saved.get("api_key"):
        return
    if all(saved.get(key) == value for key, value in fields.items()):
        return
    saved.update(fields)
    _write_private(credentials_path(), saved)


def stored_api_key() -> str | None:
    """The key saved by ``wai login``, or ``None``."""
    data = _read_credentials()
    key = (data or {}).get("api_key")
    return str(key) if key else None


def resolve_api_key(explicit: str | None = None) -> str | None:
    """``explicit`` > ``wai.configure(api_key=)`` > ``WHILEAI_API_KEY`` > the saved credentials file."""
    from .config import current

    return explicit or current().api_key or getenv("API_KEY") or stored_api_key()


def _post(path: str, body: dict, timeout: int = 30) -> tuple[int, dict]:
    request = urllib.request.Request(
        _api_url() + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as err:
        raw = err.read().decode(errors="replace")
        try:
            return err.code, json.loads(raw)
        except ValueError:
            return err.code, {"error": raw[:200]}
    except urllib.error.URLError as err:
        raise LoginError(f"Could not reach {_api_url()}: {err.reason}") from None
    except OSError as err:  # socket timeout, reset, DNS blip
        raise LoginError(f"Could not reach {_api_url()}: {err}") from None


def _get(path: str, api_key: str, timeout: int = 30) -> tuple[int, dict]:
    request = urllib.request.Request(
        _api_url() + path, method="GET", headers={"X-Api-Key": api_key}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as err:
        raw = err.read().decode(errors="replace")
        try:
            return err.code, json.loads(raw)
        except ValueError:
            return err.code, {"error": raw[:200]}
    except OSError as err:
        raise LoginError(f"Could not reach {_api_url()}: {err}") from None


def account(api_key: str | None = None) -> dict:
    """Tier, limits and today's usage for the key in use (``GET /me``).

    ``tier`` is ``"trial"`` for an account made by ``signup`` that has not
    signed in yet; ``trial["lift"]`` says how to lift it.
    """
    key = resolve_api_key(api_key)
    if not key:
        raise LoginError("No key. Run `wai login` or `wai signup --email`.")
    status, data = _get("/me", key)
    if status != 200:
        raise LoginError(f"Account lookup failed ({status}): {data.get('error', data)}")
    return data


def _default_name() -> str:
    host = socket.gethostname().split(".")[0] or "cli"
    return f"cli {host}"[:50]


def _start(name: str) -> dict:
    status, flow = _post("/device/code", {"name": name})
    if status != 200 or "device_code" not in flow:
        raise LoginError(f"Login could not start ({status}): {flow.get('error', flow)}")
    flow["api_url"] = _api_url()
    flow["expires_at"] = time.time() + int(flow.get("expires_in", 900))
    return flow


def _resume() -> dict | None:
    flow = _read(_pending_path())
    if not flow or flow.get("api_url") != _api_url():
        return None
    if float(flow.get("expires_at", 0)) - 30 <= time.time():
        return None
    return flow


def login(
    *,
    name: str | None = None,
    wait: bool = True,
    timeout: float | None = None,
    open_browser: bool = True,
    out: Callable[[str], None] | None = None,
) -> str | None:
    """Run the device login. Returns the API key, or ``None`` if still pending.

    ``wait=False`` prints the link and returns at once; run again to finish.
    ``timeout`` caps the wait in seconds (default: until the code expires).
    """
    say = out or (lambda s: print(s, file=sys.stderr, flush=True))
    flow = _resume()
    if flow is None:
        flow = _start(name or _default_name())
        _write_private(_pending_path(), flow)
        say("Open this link and press Approve:")
    else:
        say("Resuming the login you started. Open this link and press Approve:")
    say("")
    say(f"    {flow['verification_uri_complete']}")
    say("")
    say(f"    code: {flow['user_code']}")
    say("")
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(flow["verification_uri_complete"])
    if not wait:
        say("Run `wai login` again once you have approved.")
        return None

    deadline = float(flow["expires_at"])
    if timeout is not None:
        deadline = min(deadline, time.time() + timeout)
    interval = max(1, int(flow.get("interval", 5)))
    body = {"device_code": flow["device_code"], "user_code": flow["user_code"]}
    say("Waiting for approval...")
    failures = 0
    while True:
        try:
            status, data = _post("/device/token", body)
        except LoginError:
            failures += 1
            if failures > 5:
                raise
            time.sleep(interval)
            continue
        failures = 0
        if status == 200 and data.get("api_key"):
            _write_private(
                credentials_path(),
                {
                    "api_key": data["api_key"],
                    "api_url": flow["api_url"],
                    "name": data.get("name"),
                    "user_id": data.get("user_id"),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            with contextlib.suppress(OSError):
                _pending_path().unlink()
            # the tier the key carries, saved next to it, so a run can
            # name a trial limit before it spends one
            with contextlib.suppress(LoginError, OSError, ValueError):
                remember_account(account(data["api_key"]))
            say(f"Logged in. Key saved to {credentials_path()}")
            return data["api_key"]
        error = data.get("error", "")
        if error == "authorization_pending":
            if time.time() >= deadline:
                say("Still waiting. Run `wai login` again to keep waiting.")
                return None
            time.sleep(interval)
            continue
        if error == "slow_down":
            interval += 5
            time.sleep(interval)
            continue
        with contextlib.suppress(OSError):
            _pending_path().unlink()
        if error == "expired_token":
            raise LoginError("That code expired. Run `wai login` again.")
        raise LoginError(f"Login failed ({status}): {error or data}")


def signup(email: str, *, name: str | None = None, out: Callable[[str], None] | None = None) -> str:
    """Create an account for ``email`` and save its API key. Returns the key.

    No browser and no password: the person opens the dashboard later by
    signing in with an email code. Raises ``LoginError`` if the address
    already has an account (run ``login`` instead).
    """
    say = out or (lambda s: print(s, file=sys.stderr, flush=True))
    email = str(email or "").strip()
    if "@" not in email:
        raise LoginError("Pass a valid email address.")
    status, data = _post("/signup", {"email": email, "name": name or _default_name()})
    if status == 201 and data.get("api_key"):
        _write_private(
            credentials_path(),
            {
                "api_key": data["api_key"],
                "api_url": _api_url(),
                "name": data.get("name"),
                "user_id": data.get("user_id"),
                "email": data.get("email", email),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # so simulate() can warn about the trial before a hosted
                # run spends it, without a call to /me
                **_tier_fields(data),
            },
        )
        say(f"Account created for {data.get('email', email)}. Key saved to {credentials_path()}")
        trial = data.get("trial") or {}
        if data.get("tier") == "trial":
            say(
                "Trial key: "
                f"{trial.get('daily_input_tokens', 25000):,} input / {trial.get('daily_output_tokens', 50000):,} "
                f"output tokens a day, {int(trial.get('storage_bytes', 104857600)) // 1048576} MB, "
                f"{trial.get('datasets', 10)} datasets, expires {str(trial.get('expires_at', ''))[:10]}."
            )
            say(trial_note(trial.get("daily_input_tokens")))
            say(
                trial.get("lift")
                or f"Sign in once at {SIGN_IN_URL} with an email code to lift trial limits."
            )
        else:
            say(f"Platform: sign in at {SITE_URL} with an email code.")
        return data["api_key"]
    error = data.get("error", "")
    if error == "account_exists":
        raise LoginError(f"{email} already has an account. Run `wai login`.")
    if error == "invalid_email":
        raise LoginError("Pass a valid email address.")
    if error == "too_many_signups":
        raise LoginError("Too many sign-ups from this network today. Try again tomorrow.")
    raise LoginError(f"Sign-up failed ({status}): {error or data}")


def logout() -> bool:
    """Delete the saved credentials. Returns whether anything was removed."""
    removed = False
    for path in (credentials_path(), _pending_path()):
        try:
            path.unlink()
            removed = removed or path.name == "credentials.json"
        except OSError:
            pass
    return removed


def status() -> dict:
    """What the SDK would use right now, with the key masked.

    On a trial key ``trial_note`` says how many hosted situations the
    daily allowance covers and names the offline writer that has none.
    ``tier`` comes from ``GET /me``, so reading it costs one call.
    """
    env = getenv("API_KEY")
    env_var = env_name("API_KEY")
    saved = _read_credentials() or {}
    key = env or saved.get("api_key")
    out = {
        "api_url": _api_url(),
        "configured": bool(key),
        "source": env_var if env else ("file" if saved.get("api_key") else None),
        "path": str(credentials_path()),
        "key": (key[:7] + "..." + key[-4:]) if key and len(key) > 12 else (key or None),
        "name": None if env else saved.get("name"),
        "pending": _resume() is not None,
        "tier": None,
        # where a hosted simulate/grade goes: the shared pool on VLLM_API_KEY,
        # else the account endpoints on this key, else nowhere until one exists
        "hosted_route": (
            "shared pool (VLLM_API_KEY)"
            if os.environ.get("VLLM_API_KEY", "").strip()
            else ("account endpoints (this key)" if key else None)
        ),
    }
    if key:
        try:
            me = account(key)
        except LoginError as err:
            out["tier"] = f"unknown ({err})"
        else:
            out["tier"] = me.get("tier")
            if not env:
                remember_account(me)
            if me.get("tier") == "trial":
                trial = me.get("trial") or {}
                out["trial_expires_at"] = trial.get("expires_at")
                out["lift"] = trial.get("lift")
                out["trial_note"] = trial_note(trial.get("daily_input_tokens"))
    return out

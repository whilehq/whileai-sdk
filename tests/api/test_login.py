"""`wai login`: the device flow from the CLI's side, with the gate faked."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from whileai import auth, cli
from whileai.simulations.ingest import platform


class FakeGate:
    """Records the calls the CLI makes and answers like the token gate."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.approved = False
        self.started = 0

    def post(self, path, body, timeout=30):
        self.calls.append((path, body))
        if path == "/device/code":
            self.started += 1
            return 200, {
                "device_code": "d" * 64,
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://while.ai/device",
                "verification_uri_complete": "https://while.ai/device?code=ABCD-EFGH",
                "expires_in": 900,
                "interval": 0,
            }
        if path == "/device/token":
            assert body == {"device_code": "d" * 64, "user_code": "ABCD-EFGH"}
            if self.approved:
                return 200, {"api_key": "zp_" + "a" * 48, "name": "cli box", "user_id": "user_1"}
            return 400, {"error": "authorization_pending"}
        if path == "/signup":
            if body["email"] == "taken@example.com":
                return 409, {"error": "account_exists"}
            return 201, {
                "api_key": "zp_" + "b" * 48,
                "name": body["name"],
                "user_id": "user_new",
                "email": body["email"].lower(),
                "tier": "trial",
                "trial": TRIAL,
            }
        raise AssertionError(path)

    def get(self, path, api_key, timeout=30):
        self.calls.append((path, {"api_key": api_key}))
        assert path == "/me"
        if api_key == "zp_" + "b" * 48:
            return 200, {
                "user_id": "user_new",
                "tier": "trial",
                "trial": TRIAL,
                "limits": {"dailyInputTokens": 25000},
                "usage": {"inputTokens": 0},
            }
        if api_key.startswith("zp_"):
            return 200, {
                "user_id": "user_1",
                "tier": "full",
                "trial": None,
                "limits": {},
                "usage": {},
            }
        return 401, {"error": "Invalid API key"}


TRIAL = {
    "expires_at": "2026-09-21T00:00:00.000Z",
    "daily_input_tokens": 25000,
    "daily_output_tokens": 50000,
    "storage_bytes": 104857600,
    "datasets": 10,
    "lift": f"Sign in once at {auth.SIGN_IN_URL} with an email code to lift trial limits.",
}


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path))
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.delenv("WHILEAI_DELEGATED_CREDENTIAL", raising=False)
    monkeypatch.delenv("WHILEAI_API_URL", raising=False)
    fake = FakeGate()
    monkeypatch.setattr(auth, "_post", fake.post)
    monkeypatch.setattr(auth, "_get", fake.get)
    monkeypatch.setattr(auth.time, "sleep", lambda s: None)
    return fake


def test_login_prints_link_and_saves_key_after_approval(gate, tmp_path):
    lines: list[str] = []
    polls = {"n": 0}

    def approve_on_second_poll(path, body, timeout=30):
        if path == "/device/token":
            polls["n"] += 1
            gate.approved = polls["n"] >= 2
        return FakeGate.post(gate, path, body, timeout)

    auth._post = approve_on_second_poll
    key = auth.login(open_browser=False, out=lines.append)

    assert key == "zp_" + "a" * 48
    text = "\n".join(lines)
    assert "https://while.ai/device?code=ABCD-EFGH" in text
    assert "ABCD-EFGH" in text
    saved = json.loads((tmp_path / "credentials.json").read_text())
    assert saved["api_key"] == key
    assert saved["name"] == "cli box"
    assert saved["tier"] == "full", "login records the tier it learned from /me"
    assert auth.trial_prerun_note() is None
    assert saved["api_url"] == auth.DEFAULT_API_URL
    assert not (tmp_path / "pending-login.json").exists()
    assert auth.stored_api_key() == key
    assert auth.resolve_api_key() == key


def test_no_wait_then_resume_uses_the_same_code(gate, tmp_path):
    assert auth.login(wait=False, open_browser=False, out=lambda s: None) is None
    assert (tmp_path / "pending-login.json").exists()
    assert gate.started == 1

    gate.approved = True
    lines: list[str] = []
    key = auth.login(open_browser=False, out=lines.append)
    assert key
    assert gate.started == 1, "resumed instead of minting a new code"
    assert "Resuming" in lines[0]


def test_timeout_leaves_the_pending_login_for_next_time(gate, tmp_path):
    assert auth.login(open_browser=False, timeout=0, out=lambda s: None) is None
    assert (tmp_path / "pending-login.json").exists()
    assert auth.status()["pending"] is True


def test_expired_pending_login_starts_over(gate, tmp_path):
    auth.login(wait=False, open_browser=False, out=lambda s: None)
    pending = json.loads((tmp_path / "pending-login.json").read_text())
    pending["expires_at"] = time.time() - 1
    (tmp_path / "pending-login.json").write_text(json.dumps(pending))
    gate.approved = True
    assert auth.login(open_browser=False, out=lambda s: None)
    assert gate.started == 2


def test_expired_code_at_the_gate_is_a_clear_error(gate, tmp_path):
    def expired(path, body, timeout=30):
        if path == "/device/token":
            return 400, {"error": "expired_token"}
        return FakeGate.post(gate, path, body, timeout)

    auth._post = expired
    with pytest.raises(auth.LoginError, match="expired"):
        auth.login(open_browser=False, out=lambda s: None)
    assert not (tmp_path / "pending-login.json").exists()


def test_transient_network_errors_while_waiting_are_retried(gate):
    blips = {"left": 2}

    def flaky(path, body, timeout=30):
        if path == "/device/token" and blips["left"]:
            blips["left"] -= 1
            raise auth.LoginError("Could not reach the gate: timed out")
        gate.approved = True
        return FakeGate.post(gate, path, body, timeout)

    auth._post = flaky
    assert auth.login(open_browser=False, out=lambda s: None)
    assert blips["left"] == 0


def test_persistent_network_failure_gives_up(gate):
    def down(path, body, timeout=30):
        if path == "/device/token":
            raise auth.LoginError("Could not reach the gate: timed out")
        return FakeGate.post(gate, path, body, timeout)

    auth._post = down
    with pytest.raises(auth.LoginError, match="Could not reach"):
        auth.login(open_browser=False, out=lambda s: None)


def test_key_name_defaults_to_the_host(gate):
    auth.login(wait=False, open_browser=False, out=lambda s: None)
    name = gate.calls[0][1]["name"]
    assert name.startswith("cli ") and len(name) <= 50
    auth.logout()
    auth.login(wait=False, name="claude-code", open_browser=False, out=lambda s: None)
    assert gate.calls[-1][1]["name"] == "claude-code"


def test_platform_calls_fall_back_to_the_saved_key(gate, monkeypatch):
    with pytest.raises(platform.PlatformError, match="wai login"):
        platform._key(None)
    gate.approved = True
    auth.login(open_browser=False, out=lambda s: None)
    assert platform._key(None) == "zp_" + "a" * 48
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_env")
    assert platform._key(None) == "zp_env", "the env var still wins"
    assert platform._key("zp_explicit") == "zp_explicit"


def test_logout_and_status(gate, capsys):
    assert auth.logout() is False
    gate.approved = True
    assert cli.main(["login", "--no-browser"]) == 0
    assert cli.main(["status"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["source"] == "file"
    assert shown["key"] == "zp_aaaa...aaaa"
    assert shown["name"] == "cli box"
    assert shown["tier"] == "full"
    assert "lift" not in shown
    assert cli.main(["logout"]) == 0
    assert auth.stored_api_key() is None
    assert cli.main(["status"]) == 0
    # after a logout, status is JSON plus the keyless note (#790), not an error
    out = capsys.readouterr().out.split("Logged out.\n")[-1]
    body, note = out.rsplit("\n}\n", 1)
    assert json.loads(body + "\n}")["source"] is None
    assert note.startswith("no API key: the library runs without one")


def test_status_records_the_tier_on_an_older_credentials_file(gate, tmp_path):
    gate.approved = True
    auth.login(open_browser=False, out=lambda s: None)
    path = tmp_path / "credentials.json"
    stale = json.loads(path.read_text())
    stale.pop("tier", None)
    path.write_text(json.dumps(stale))
    assert auth.trial_prerun_note() is None

    assert auth.status()["tier"] == "full"
    assert json.loads(path.read_text())["tier"] == "full"


def test_signup_creates_the_account_and_saves_the_key(gate, tmp_path):
    lines: list[str] = []
    key = auth.signup("Agent@Example.com", name="claude-code", out=lines.append)
    assert key == "zp_" + "b" * 48
    saved = json.loads((tmp_path / "credentials.json").read_text())
    assert saved["email"] == "agent@example.com"
    assert saved["user_id"] == "user_new"
    # the tier rides along, so a run can name the trial limit before it
    # spends one without asking /me
    assert saved["tier"] == "trial"
    assert saved["daily_input_tokens"] == 25000
    assert saved["expires_at"] == "2026-09-21T00:00:00.000Z"
    assert auth.trial_prerun_note().startswith("trial key: the hosted writer covers about 12")
    assert gate.calls[-1] == ("/signup", {"email": "Agent@Example.com", "name": "claude-code"})
    assert "Account created" in lines[0]
    assert "Trial key" in lines[1] and "25,000" in lines[1] and "2026-09-21" in lines[1]
    assert "sign-in" in lines[2]
    assert platform._key(None) == key
    me = auth.account()
    assert me["tier"] == "trial"
    assert me["trial"]["lift"].startswith("Sign in once")
    shown = auth.status()
    assert shown["tier"] == "trial"
    assert shown["trial_expires_at"] == "2026-09-21T00:00:00.000Z"
    assert "sign-in" in shown["lift"]


def test_signup_existing_account_points_at_login(gate):
    with pytest.raises(auth.LoginError, match="wai login"):
        auth.signup("taken@example.com", out=lambda s: None)
    with pytest.raises(auth.LoginError, match="valid email"):
        auth.signup("nope", out=lambda s: None)
    assert auth.stored_api_key() is None


def test_cli_signup(gate, capsys):
    assert cli.main(["signup", "--email", "new@example.com"]) == 0
    shown = auth.status()
    assert shown["source"] == "file"
    assert shown["tier"] == "trial"
    assert cli.main(["signup", "--email", "taken@example.com"]) == 1
    assert "wai login" in capsys.readouterr().err


def test_cli_exit_codes(gate):
    assert cli.main(["login", "--no-browser", "--no-wait"]) == 0
    assert cli.main(["login", "--no-browser", "--timeout", "0"]) == 2

    def down(path, body, timeout=30):
        raise auth.LoginError("Could not reach the gate")

    auth._post = down
    auth.logout()
    assert cli.main(["login", "--no-browser"]) == 1


def test_config_dir_ignores_a_leftover_zeroproof_home(tmp_path, monkeypatch):
    """``~/.zeroproof`` from the package's old name is no longer consulted, and
    neither is ``$ZEROPROOF_HOME``: ``$WHILEAI_HOME``, else ``~/.whileai``."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("WHILEAI_HOME", raising=False)
    monkeypatch.setenv("ZEROPROOF_HOME", str(tmp_path / "elsewhere"))
    old = tmp_path / ".zeroproof"
    old.mkdir()
    (old / "credentials.json").write_text(json.dumps({"api_key": "zp_old"}), encoding="utf-8")
    assert auth.config_dir() == tmp_path / ".whileai"
    assert auth.stored_api_key() is None
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path / "home"))
    assert auth.config_dir() == tmp_path / "home"


def test_login_saved_against_the_retired_gate_host_moves_to_the_default_api(tmp_path, monkeypatch):
    """A credentials file that pinned api.zeroproofai.com (the token gate the
    login moved off) still counts as a login and reads as the default API."""
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path))
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    (tmp_path / "credentials.json").write_text(
        json.dumps({"api_key": "zp_" + "c" * 48, "api_url": "https://api.zeroproofai.com"}),
        encoding="utf-8",
    )
    assert auth.stored_api_key() == "zp_" + "c" * 48
    assert auth._read_credentials()["api_url"] == auth.DEFAULT_API_URL
    # only that exact host is rewritten; a staging gate someone pinned stays
    (tmp_path / "credentials.json").write_text(
        json.dumps({"api_key": "zp_" + "c" * 48, "api_url": "https://gate.staging.example"}),
        encoding="utf-8",
    )
    assert auth._read_credentials()["api_url"] == "https://gate.staging.example"

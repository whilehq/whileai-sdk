"""Two things that lock a new user out inside their first hour.

Both fail on the commit before the fix: a user's own Modal endpoint
demanding our account key, and `wai login` printing a link that 404s in a
signed-out browser.
"""

from __future__ import annotations

from whileai import auth
from whileai._env import PLATFORM_MODAL_PREFIX, is_platform_host
from whileai.simulations.generate.agents import _hosted_qwen_url, missing_hosted_key

# --------------------------------------------------------------- own endpoint

#: A user serving their own Qwen3-4B on their own L40S, on their own Modal
#: account. Nothing about it is ours but the four letters at the end.
MY_OWN_MODAL = "https://mylab--my-qwen-vllm-serve.modal.run/v1"
#: What While actually serves from. Built from the prefix ``is_platform_host``
#: carries, so this file never spells the retired name itself.
WHILES_OWN_MODAL = f"https://{PLATFORM_MODAL_PREFIX}qwen3-4b.modal.run/v1"


def test_my_own_modal_endpoint_is_not_whiles_hosted_model():
    """`modal.run` is a landlord, not an owner. CONSTITUTION.md §4."""
    assert not is_platform_host(MY_OWN_MODAL)
    assert not _hosted_qwen_url(MY_OWN_MODAL), (
        "a user's own Modal app was classified as While's hosted model"
    )
    assert is_platform_host(WHILES_OWN_MODAL)
    assert _hosted_qwen_url(WHILES_OWN_MODAL), "While's own endpoint must stay hosted"


def test_my_own_modal_endpoint_is_never_asked_for_an_account_key(monkeypatch):
    """The old error told them to `wai login`, then recommended the spelling
    they had already used. Now it names a variable that actually reaches it."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    err = missing_hosted_key(MY_OWN_MODAL)
    assert err, "an endpoint with no key at all still has to say so"
    assert "wai login" not in err and "account key" not in err, (
        f"a self-hosted endpoint was sent to buy an account key: {err}"
    )
    assert "VLLM_API_KEY" in err, f"the error has to name a key that works: {err}"


def test_my_own_modal_endpoint_accepts_the_key_i_set(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "sk-mine")
    assert missing_hosted_key(MY_OWN_MODAL) is None


# -------------------------------------------------------------------- login


def test_login_link_is_openable_by_a_signed_out_browser():
    """`while.ai/device?code=` 404s with no session; `/sign-in?redirect=` is 200."""
    link = auth.approval_link("https://while.ai/device?code=ABCD-EFGH")
    assert link.startswith(auth.SIGN_IN_URL + "?redirect="), link
    assert "%2Fdevice" in link and "ABCD-EFGH" in link, link


def test_login_does_not_wrap_a_link_twice_or_a_host_that_is_not_ours():
    assert auth.approval_link(auth.SIGN_IN_URL + "?redirect=%2Fdevice") == (
        auth.SIGN_IN_URL + "?redirect=%2Fdevice"
    )
    mine = "https://gate.example.com/device?code=ABCD-EFGH"
    assert auth.approval_link(mine) == mine, "a self-hosted gate's link is printed as it came"


def test_wai_login_prints_the_signed_out_link(monkeypatch, tmp_path):
    """The first command in `wai --help` was the first one to fail."""
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path))
    monkeypatch.delenv("WHILEAI_API_URL", raising=False)
    state = {"polls": 0}

    def post(path, body, timeout=30):
        if path == "/device/code":
            return 200, {
                "device_code": "d" * 64,
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://while.ai/device",
                "verification_uri_complete": "https://while.ai/device?code=ABCD-EFGH",
                "expires_in": 900,
                "interval": 0,
            }
        if path == "/device/token":
            state["polls"] += 1
            if state["polls"] >= 2:
                return 200, {"api_key": "zp_" + "a" * 48, "name": "cli box", "user_id": "u1"}
            return 400, {"error": "authorization_pending"}
        raise AssertionError(path)

    def get(path, api_key, timeout=30):
        return 200, {"user_id": "u1", "tier": "full", "trial": None, "limits": {}, "usage": {}}

    monkeypatch.setattr(auth, "_post", post)
    monkeypatch.setattr(auth, "_get", get)
    monkeypatch.setattr(auth.time, "sleep", lambda s: None)
    lines: list[str] = []
    auth.login(open_browser=False, out=lines.append)
    text = "\n".join(lines)
    assert "/sign-in?redirect=" in text, f"login printed a bare protected URL: {text}"
    assert "https://while.ai/device?code=ABCD-EFGH" not in text

"""``wai.platform.hosted``: the hosted endpoint's client. No live calls."""

from __future__ import annotations

import pytest

import whileai as wai
from whileai.hosted import HostedModel, HostedModels, Subdomain, UsageDay, models_url
from whileai.platform import PlatformError

ARN = "arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123def456"
ROLE = "arn:aws:iam::123456789012:role/WhileModelsInvoke"


class Fake:
    """Records every call; answers like models.withwhile.com."""

    def __init__(self):
        self.calls: list[tuple[str, str, object]] = []
        self.rows: dict[str, dict] = {}
        self.slug: str | None = None

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/models" and method == "POST":
            row = {**body, "createdAt": "2026-09-21T00:00:00Z", "updatedAt": "2026-09-21T00:00:00Z"}
            if row.get("kind") == "bedrock":
                row.setdefault("region", "us-east-1")
                if "roleArn" in row:
                    row.setdefault("externalId", "while-models")
            else:
                row["upstreamModel"] = row.pop("model")
            self.rows[body["name"]] = row
            return {"model": row, "endpoint": "https://models.withwhile.com/v1"}
        if path == "/models" and method == "GET":
            return {"models": list(self.rows.values())}
        if path.startswith("/models/") and path.endswith("/usage?days=3"):
            name = path.split("/")[2]
            return {
                "usage": [
                    {
                        "day": "2026-09-21",
                        "model": name,
                        "calls": 12,
                        "errors": 1,
                        "inputTokens": 300,
                        "outputTokens": 40,
                    }
                ]
            }
        if path.startswith("/models/") and method == "GET":
            name = path.split("/")[2]
            row = self.rows.get(name)
            if row and row.get("status") == "importing":
                self.polls = getattr(self, "polls", 0) + 1
                if getattr(self, "fail_import", False):
                    row.update(
                        status="failed", step="failed", error="Bedrock import failed: bad shard"
                    )
                elif self.polls >= 2:
                    row.update(status="ready", step="ready", arn=ARN, cmu=2)
            if name not in self.rows:
                raise PlatformError(404, f"GET {path}: No model named {name}")
            return {"model": self.rows[name]}
        if path.startswith("/models/") and method == "DELETE":
            return {"deleted": self.rows.pop(path.split("/")[2])["name"]}
        if path == "/domain":
            if method == "PUT":
                self.slug = body["subdomain"]
            if method == "DELETE":
                released, self.slug = self.slug, None
                return {"released": released}
            s = self.slug
            return {
                "subdomain": s,
                "host": s and f"{s}.models.withwhile.com",
                "endpoint": s and f"https://{s}.models.withwhile.com/v1",
            }
        if path == "/models/import" and method == "POST":
            row = {
                "name": body["name"],
                "kind": "bedrock",
                "status": "importing",
                "step": "queued",
                "region": "us-east-1",
            }
            self.rows[body["name"]] = row
            self.imports = getattr(self, "imports", 0) + 1
            return {"model": row, "status": "importing"}
        raise AssertionError(f"unexpected {method} {path}")


def test_register_a_bedrock_model_in_your_own_account():
    fake = Fake()
    m = HostedModels(transport=fake).register(
        "t2s", arn=ARN, role_arn=ROLE, base="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    )
    assert isinstance(m, HostedModel)
    method, path, body = fake.calls[0]
    assert (method, path) == ("POST", "/models")
    assert body == {
        "name": "t2s",
        "kind": "bedrock",
        "arn": ARN,
        "roleArn": ROLE,
        "base": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    }
    assert (
        m.kind == "bedrock"
        and m.region == "us-east-1"
        and m.role_arn == ROLE
        and m.external_id == "while-models"
    )
    assert m.endpoint == "https://models.withwhile.com/v1"
    assert "model='t2s'" in str(m) and ARN in str(m)


def test_register_an_openai_compatible_server():
    fake = Fake()
    m = HostedModels(transport=fake).register(
        "qwen", url="https://x.modal.run/v1", model="acme/qwen/v3", auth="none"
    )
    assert fake.calls[0][2] == {
        "name": "qwen",
        "kind": "openai",
        "url": "https://x.modal.run/v1",
        "model": "acme/qwen/v3",
        "auth": "none",
    }
    assert m.kind == "openai" and m.upstream_model == "acme/qwen/v3" and m.auth == "none"


def test_register_refuses_an_ambiguous_or_incomplete_call():
    h = HostedModels(transport=Fake())
    with pytest.raises(ValueError, match="exactly one of arn="):
        h.register("m")
    with pytest.raises(ValueError, match="exactly one of arn="):
        h.register("m", arn=ARN, url="https://h/v1", model="x")
    with pytest.raises(ValueError, match="also needs model="):
        h.register("m", url="https://h/v1")


def test_list_get_delete_and_usage():
    fake = Fake()
    h = HostedModels(transport=fake)
    h.register("a", arn=ARN)
    h.register("b", url="https://h/v1", model="m")
    assert [m.name for m in h.list()] == ["a", "b"]
    assert h.get("a").arn == ARN
    with pytest.raises(PlatformError, match="No model named zzz"):
        h.get("zzz")
    days = h.usage("a", days=3)
    assert days == [
        UsageDay(
            day="2026-09-21", model="a", calls=12, errors=1, input_tokens=300, output_tokens=40
        )
    ]
    h.delete("a")
    assert [m.name for m in h.list()] == ["b"]
    assert ("DELETE", "/models/a", None) in fake.calls


def test_subdomain_claim_read_and_release():
    fake = Fake()
    h = HostedModels(transport=fake)
    assert str(h.subdomain()) == "no subdomain claimed"
    s = h.subdomain("acme")
    assert isinstance(s, Subdomain)
    assert s.endpoint == "https://acme.models.withwhile.com/v1"
    assert h.subdomain().subdomain == "acme"
    assert h.release_subdomain() == "acme"
    assert h.subdomain().subdomain is None


def test_endpoint_is_a_wai_endpoint_on_the_account_key(monkeypatch):
    monkeypatch.setenv("WHILEAI_API_KEY", "zp_test")
    e = HostedModels(transport=Fake()).endpoint("t2s")
    assert isinstance(e, wai.Endpoint)
    assert e.spec == "vllm:t2s@https://models.withwhile.com/v1"
    assert e.api_key == "zp_test"


def test_models_url_is_overridable(monkeypatch):
    monkeypatch.setenv("WHILEAI_MODELS_URL", "https://staging.example/")
    assert models_url() == "https://staging.example"
    monkeypatch.delenv("WHILEAI_MODELS_URL")
    assert models_url() == "https://models.withwhile.com"


def test_the_namespace_hangs_off_platform():
    assert isinstance(wai.platform.hosted, HostedModels)


def test_publish_hands_over_an_adapter_and_waits_for_ready():
    fake = Fake()
    h = HostedModels(transport=fake)
    m = h.publish("while-ai/airline-concise-4b", hf_token="hf_once", poll_s=0)
    method, path, body = fake.calls[0]
    assert (method, path) == ("POST", "/models/import")
    assert body == {
        "name": "airline-concise-4b",
        "adapter": "while-ai/airline-concise-4b",
        "hfToken": "hf_once",
    }
    assert m.status == "ready" and m.arn == ARN and m.cmu == 2
    assert fake.polls == 2


def test_publish_without_wait_returns_the_importing_row():
    fake = Fake()
    m = HostedModels(transport=fake).publish(
        "run_f69e975a1571d445", name="t2s", base="nvidia/Llama-3.1-Nemotron-Nano-8B-v1", wait=False
    )
    assert m.status == "importing" and m.step == "queued"
    assert fake.calls[0][2]["base"] == "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"


def test_publish_raises_when_the_import_fails():
    fake = Fake()
    fake.fail_import = True
    with pytest.raises(RuntimeError, match="t2s failed to import: Bedrock import failed"):
        HostedModels(transport=fake).publish("run_f69e975a1571d445", name="t2s", poll_s=0)

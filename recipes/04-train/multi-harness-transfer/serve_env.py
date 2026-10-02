"""Start FineEnvs' SmolDataEnvs Harbor server with the capture proxy published by `modal.forward`.

OpenEnv ships `gradio`, `cloudflare` and `direct` forwarders. Inside a Modal container the
native answer is a Modal tunnel, so this registers one more forwarder before the server imports.
"""

import os

import modal
import uvicorn
from openenv.core.harness.capture import forwarding


class ModalForwarder(forwarding.PortForwarder):
    def __init__(self) -> None:
        super().__init__()
        self._ctx = None

    def start(self, local_port: int, *, local_host: str = "127.0.0.1") -> str:
        self._local_port = local_port
        self._ctx = modal.forward(local_port)
        self._url = self._ctx.__enter__().url
        return self._url

    def stop(self) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None
        self._url = None


forwarding._FORWARDERS["modal"] = ModalForwarder
os.environ["OPENENV_EXPOSE"] = "modal"


def install_shuffle() -> None:
    """Phase 1 arm C only (HARNESS_SHUFFLE=1): rewrite tool spelling per capture session.

    The proxy resolves the session, then calls the module-level `normalise_for_capture` on the
    chat request and `normalise_response` on the engine's answer. Both are looked up at call time,
    so wrapping them here is enough; a context variable carries the session between the two.
    """
    import contextvars

    from openenv.core.harness.capture import server as cap
    from openenv.core.harness.capture import sessions

    from harness_shuffle import draw, restore_response, rewrite_request

    current: contextvars.ContextVar = contextvars.ContextVar("shuffle_session", default=None)
    spellings: dict = {}

    resolve = sessions.SessionRegistry.resolve

    def resolve_and_remember(self, *a, **kw):
        session = resolve(self, *a, **kw)
        current.set(getattr(session, "session_id", None))
        return session

    sessions.SessionRegistry.resolve = resolve_and_remember

    before, after = cap.normalise_for_capture, cap.normalise_response

    def before_engine(chat_request):
        before(chat_request)
        sid = current.get()
        if sid and chat_request.get("tools"):
            spellings.setdefault(sid, draw(sid, chat_request["tools"]))
        if sid in spellings:
            rewrite_request(chat_request, spellings[sid])

    def after_engine(response):
        after(response)
        sid = current.get()
        if sid in spellings:
            restore_response(response, spellings[sid])

    cap.normalise_for_capture, cap.normalise_response = before_engine, after_engine


if os.environ.get("HARNESS_SHUFFLE") == "1":
    install_shuffle()

if __name__ == "__main__":
    from smoldataenv_harbor.server import app

    uvicorn.run(app, host="127.0.0.1", port=8200, ws_ping_timeout=1800, log_level="warning")

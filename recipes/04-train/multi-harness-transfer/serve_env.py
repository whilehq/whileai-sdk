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

if __name__ == "__main__":
    from smoldataenv_harbor.server import app

    uvicorn.run(app, host="127.0.0.1", port=8200, ws_ping_timeout=1800, log_level="warning")

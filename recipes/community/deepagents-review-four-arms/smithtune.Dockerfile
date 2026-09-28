# smithtune 0.1.0 imports fcntl, so it needs Linux or macOS. This image is the
# same install its README gives, pinned, for running it from Windows:
#   docker build -f smithtune.Dockerfile -t smithtune:0.1.0 .
#   docker run --rm -e LANGSMITH_API_KEY -e BASETEN_API_KEY -v "$PWD/.cache:/work" -w /work smithtune:0.1.0 smithtune doctor
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.16

ENV UV_TOOL_BIN_DIR=/usr/local/bin
RUN uv tool install --python 3.12 \
    --overrides https://raw.githubusercontent.com/langchain-ai/smithtune/v0.1.0/overrides.txt \
    'smithtune[deepagents] @ git+https://github.com/langchain-ai/smithtune.git@v0.1.0'

# The LangSmith CLI smithtune shells out to for dataset pull/push.
RUN curl -fsSL https://github.com/langchain-ai/langsmith-cli/releases/download/v0.2.59/langsmith_linux_amd64.tar.gz \
    | tar -xz -C /usr/local/bin langsmith

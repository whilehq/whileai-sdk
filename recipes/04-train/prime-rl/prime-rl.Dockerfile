# Prime Intellect's published prime-rl image, with the python symlink Modal will
# re-create cleared (see modal_prime_rl.py). PRIME_RL_IMAGE is passed as a build arg.
ARG PRIME_RL_IMAGE=ghcr.io/primeintellect-ai/prime-rl:v0.8.1.dev63
FROM ${PRIME_RL_IMAGE}
USER root
RUN rm -f /usr/local/bin/python /usr/local/bin/python3

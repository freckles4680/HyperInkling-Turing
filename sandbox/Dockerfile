# A sandbox for Claude Code on the machine with the GPU: the agent edits, builds, tests and
# benchmarks this repository with nothing else of the host in reach. See docs/claude-sandbox.md.
#
#   sandbox/setup.sh          # writes sandbox/.env, builds this image, checks GPU and firewall
#
# The image holds no Python, CUDA or vLLM of its own. The project's venv/ was made on the host
# (uv's Python 3.14, the CUDA toolkit in venv/cuda_home), with absolute paths inside it, so
# compose.yml mounts the project, uv's Python and the toolkit at the same paths as on the host.
# BASE is the host's distribution (setup.sh reads /etc/os-release): the vLLM extension built on
# the host then finds the same glibc, libstdc++ and compiler in here.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG UID=1000
ARG GID=1000
# Extra distribution packages, e.g. gcc15-c++ on Fedora 44 (nvcc 13.3 rejects gcc 16).
ARG EXTRA_PACKAGES=""

# git, curl: the agent's own work. iptables: the firewall. iproute (ss), util-linux (taskset,
# setpriv), procps (pgrep/pkill): scripts/server-lib.sh and start.sh. gcc: Triton compiles a
# small launcher at run time; nvcc needs a host compiler. python3, jq, tmux: the scripts.
RUN set -eux; \
    if command -v dnf > /dev/null; then \
        dnf install -y --setopt=install_weak_deps=False git curl ca-certificates iptables-nft \
            iproute util-linux procps-ng findutils which less gcc gcc-c++ make python3 jq tmux \
            ${EXTRA_PACKAGES}; \
        dnf clean all; \
    else \
        export DEBIAN_FRONTEND=noninteractive; \
        apt-get update; \
        apt-get install -y --no-install-recommends git curl ca-certificates iptables iproute2 \
            util-linux procps less gcc g++ make python3 jq tmux ${EXTRA_PACKAGES}; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# The agent's user has the host user's ids, so files it writes in the project stay yours.
# (Ubuntu images ship a user "ubuntu" with uid 1000: it makes way.)
RUN set -eux; \
    if id ubuntu > /dev/null 2>&1; then userdel -r ubuntu || true; fi; \
    getent group "${GID}" > /dev/null || groupadd -g "${GID}" agent; \
    useradd -m -u "${UID}" -g "${GID}" -d /home/agent -s /bin/bash agent

# Claude Code, native installer, into /opt so that the home volume does not hide it. It does not
# update itself in here (the firewall would block it anyway): rebuild the image to update.
RUN set -eux; \
    curl -fsSL https://claude.ai/install.sh -o /tmp/install.sh; \
    HOME=/opt/claude bash /tmp/install.sh; \
    rm /tmp/install.sh; \
    ln -sf /opt/claude/.local/bin/claude /usr/local/bin/claude; \
    chmod -R a+rX /opt/claude; \
    claude --version

# Rules the agent cannot override (managed settings rank above every other settings file).
COPY managed-settings.json /etc/claude-code/managed-settings.json
COPY init-firewall.sh entrypoint.sh /usr/local/sbin/
RUN chmod 0755 /usr/local/sbin/init-firewall.sh /usr/local/sbin/entrypoint.sh

ENV HOME=/home/agent \
    PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DISABLE_AUTOUPDATER=1 \
    DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# The entrypoint runs as root only to set the firewall, then becomes the agent with no
# capabilities left; every later command goes through `docker compose exec -u agent`.
ENTRYPOINT ["/usr/local/sbin/entrypoint.sh"]
CMD ["sleep", "infinity"]

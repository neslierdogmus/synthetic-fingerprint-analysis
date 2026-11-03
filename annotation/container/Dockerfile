FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 \
    libgomp1 \
    git curl \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages (system-wide, no venv needed inside Docker) ───────────────
RUN pip3 install --no-cache-dir numpy Pillow opencv-python-headless scipy tqdm
RUN pip3 install --no-cache-dir tensorflow-cpu
RUN pip3 install --no-cache-dir pyfing==0.6

# ── Directory layout ───────────────────────────────────────────────────────────
# /opt/verifinger  → bind-mounted read-only (VeriFinger SDK, full tree)
# /data/synthetic  → bind-mounted read-only (S01-S11 synthetic sources)
# /workspace       → bind-mounted read-write, single unified mount. Everything
#                    that must survive a restart lives here: staged SDK
#                    libs/pgd, pgd.conf, license daemon storage, scripts,
#                    annotation outputs, SETUP_CHANGES.md, CONTEXT.md.
RUN mkdir -p /data/synthetic /opt/verifinger /workspace

# Reference pgd.conf — entrypoint.sh copies this into the persistent workspace
# only if a pgd.conf isn't already there, so hand-edits survive restarts. This
# file is the single source of truth for default settings.
COPY pgd.conf.default /opt/pgd_run/pgd.conf.default

WORKDIR /workspace

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

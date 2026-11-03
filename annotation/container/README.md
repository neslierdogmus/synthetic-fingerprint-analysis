> **Note for the public release:** this container was built for the original
> annotation workspace, where the scripts in `../src/` lived at
> `/workspace/annotation_src/`. Mount or copy them there. Then add your own
> Neurotechnology SDK files, which are not included here:
> `lib_linux_x86_64/` and `data/*.ndf`, plus your license setup. The scripts
> `test_license.py` and `pgd_run/` were environment-specific and are left out.

# Fingerprint Annotation Container

Annotation pipeline for synthetic fingerprint datasets.
Labels are written in the same format as real data so existing dataset classes
need no modification.

## Tools inside the container

| Task | Tool | Label file |
|------|------|------------|
| FC   | VeriFinger SDK (ctypes → libNBiometrics.so) | `{stem}.txt` |
| FMD  | VeriFinger SDK (ctypes → libNBiometrics.so) | `{stem}.mnt` |
| FOE  | pyfing SNFOE                               | `{stem}.gt`  |

See `../LABEL_FORMAT.md` for the exact byte-level format each file must match
(lives at `/workspace/LABEL_FORMAT.md`, alongside `CONTEXT.md`, not in this
directory).

## Directory layout inside the container

```
/opt/verifinger/     ← read-only: full VeriFinger SDK tree (libs, pgd, docs)
/data/synthetic/     ← read-only: S01–S11
/workspace/          ← read-write, single unified mount — everything that
                       must survive a container restart lives here
  annotation_src/
    lib_linux_x86_64/  ← SDK .so libs, staged fresh from /opt/verifinger at
                         every container start (checksum-verified)
      InferenceEngines/ ← NN inference plugins (NiemFingerNet etc.), staged
                          fresh each start too — required for FC/FMD, not
                          just the core libs (see Licensing section below)
    data/              ← SDK .ndf model/data files (Fingers.ndf etc.),
                         staged fresh each start — required for FC/FMD
    pgd_run/           ← pgd binary (also staged fresh), pgd.conf, pgd_store/
    test_license.py    ← manual license smoke test
  annotations/         ← label outputs, one subfolder per synthetic source
    S01/
    S02/
    ...
    S11/
  LicenceUsage.log     ← license server ledger — the real proof licensing
                         is server-confirmed, not just locally granted
  CONTEXT.md           ← full history, rationale, and known limitations of
                         this setup — read this before changing anything
                         licensing-related
```

The container runs **as root** (required — `pgd`, the VeriFinger license
daemon, needs root to start; this was root-caused earlier and is not
optional).

## Licensing (VeriFinger trial)

`entrypoint.sh` stages the SDK libs + `pgd` binary fresh from the read-only
`/opt/verifinger` mount into `/workspace/annotation_src/` on every container
start (checksum-verified — aborts loudly on any mismatch), starts `pgd`, and
then actively verifies the trial license was **server-confirmed** (not just
locally granted) before letting the container report ready. See
`/workspace/CONTEXT.md` for the full story, especially:
- `https = true` in `pgd.conf` is load-bearing — `https = false` silently
  produces a local-only grant with no real server confirmation.
- The boot-time check can't catch a license that silently breaks mid-project
  (only that it was confirmed at least once, ever) — **run
  `python3 /workspace/annotation_src/test_license.py` as a pre-flight check
  before kicking off any large batch run**, not just at container boot.

## SDK data files + inference engine plugins (required for FC/FMD)

A valid, server-confirmed license alone is **not** sufficient for FC
classification or FMD minutiae extraction — both also need the SDK's
`.ndf` model files (`Bin/Data/`, notably `Fingers.ndf`, the pattern
classifier model) and the neural inference-engine plugins
(`Lib/Linux_x86_64/InferenceEngines/` — `NiemFingerNet`, `NiemOrt`,
`NiemBlas`, `NiemTensorRt`, `NiemIntelIe`). Both are staged fresh from the
already-mounted `/opt/verifinger` on every boot (checksum-verified, same
principle as the lib staging) and then **functionally** verified — not
just checksummed — as part of the readiness gate, since a corrupted or
dependency-broken plugin file can pass a checksum check while still
failing to load. Missing either produces a hard failure at *use* time
(not boot time, before this fix): `"Fingers classifier model file is
missing"` (data files) or `"None of the specified Inference Engines are
available"` (plugins).

## Setup (SSH-only workflow)

### 1. Copy files to the lab machine

```bash
scp -r annotation_container/ cenglab1@<tailscale-host>:~/Desktop/synthetic_fingerprint_analysis/
```

### 2. Create the workspace folder

```bash
ssh cenglab1@<tailscale-host>
mkdir -p ~/Desktop/synthetic_fingerprint_analysis/annotation_workspace
```

This single directory is bind-mounted as `/workspace` — `annotation_src/`,
`annotations/`, `LicenceUsage.log`, and `CONTEXT.md` all live inside it and
persist across rebuilds/restarts.

### 3. Build and launch (run inside tmux in case SSH drops)

```bash
tmux new -s build
cd ~/Desktop/synthetic_fingerprint_analysis/annotation_container
bash build_and_run.sh
# Ctrl-B D to detach; tmux attach -t build to return
```

`build_and_run.sh` also warns (non-blocking) if it's not run inside tmux,
since the build pulls `tensorflow-cpu` and a dropped SSH connection can kill
a mid-build download.

### 4. Verify the container is running and actually ready

```bash
docker ps | grep fp-annotate
docker logs fp-annotate   # should end with "[entrypoint] container ready, sleeping"
```

If it stopped earlier with a `[entrypoint] FATAL: ...` line, that message
says exactly which check failed (staging checksum, pgd not starting, or
licensing not server-confirmed).

### 5. Attach with VS Code

- Open VS Code on your local machine
- Connect via **Remote SSH** to the lab machine (Tailscale host)
- Open the **Remote Explorer** sidebar → **Containers**
- Find `fp-annotate` → click **Attach to Container**
- VS Code opens a new window inside the container as **root**
- The integrated terminal is a shell at `/workspace`

### 6. Smoke test (inside the container terminal)

```bash
python3 -c "import pyfing; print('pyfing OK')"
ls /opt/verifinger/
ls /data/synthetic/
python3 /workspace/annotation_src/test_license.py   # expect: available=True
cat /workspace/LicenceUsage.log                      # expect: a FingerExtractor row
```

## Stopping and restarting

```bash
# Stop
docker stop fp-annotate

# Restart (volumes re-attached automatically, entrypoint re-stages + re-verifies)
docker start fp-annotate

# Rebuild image after Dockerfile changes, then relaunch
bash build_and_run.sh
```

## Annotation output path

All label files land next to each other under:
```
/workspace/annotations/{source_id}/{image_stem}.txt   ← FC class
/workspace/annotations/{source_id}/{image_stem}.mnt   ← FMD minutiae
/workspace/annotations/{source_id}/{image_stem}.gt    ← FOE orientation
```

The runner is skip-if-exists, so it is safe to interrupt and resume at any time.

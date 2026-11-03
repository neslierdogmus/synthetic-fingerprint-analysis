#!/bin/bash
set -euo pipefail

SDK_ARCH="Linux_x86_64"
SDK_LIB_SRC="/opt/verifinger/Lib/${SDK_ARCH}"
SDK_PGD_SRC="/opt/verifinger/Bin/${SDK_ARCH}/Activation/pgd"
SDK_DATA_SRC="/opt/verifinger/Bin/Data"
SDK_INFERENCE_SRC="/opt/verifinger/Lib/${SDK_ARCH}/InferenceEngines"

WS="/workspace/annotation_src"
LIB_DST="${WS}/lib_linux_x86_64"
DATA_DST="${WS}/data"
INFERENCE_DST="${LIB_DST}/InferenceEngines"
PGD_DIR="${WS}/pgd_run"
PGD_CONF="${PGD_DIR}/pgd.conf"
PGD_CONF_DEFAULT="/opt/pgd_run/pgd.conf.default"
USAGE_LOG="/workspace/LicenceUsage.log"
# pgd hard-codes its own operational log to /tmp/pgd.log — confirmed
# empirically across both an https=false and an https=true run, not
# controlled by pgd.conf. Do not confuse with LicenceUsageLogFile above,
# which is a different file (the server-usage ledger, our real success signal).
PGD_LOG="/tmp/pgd.log"

# ── Stage SDK libs + pgd from the read-only SDK mount into persistent storage ─
echo "[entrypoint] staging SDK libs/pgd from ${SDK_LIB_SRC} -> ${LIB_DST}"
mkdir -p "${LIB_DST}" "${PGD_DIR}/pgd_store"

for f in "${SDK_LIB_SRC}"/*.so; do
    base="$(basename "$f")"
    cp -f "$f" "${LIB_DST}/${base}.new"
    src_md5=$(md5sum "$f" | awk '{print $1}')
    dst_md5=$(md5sum "${LIB_DST}/${base}.new" | awk '{print $1}')
    if [ "$src_md5" != "$dst_md5" ]; then
        echo "[entrypoint] FATAL: checksum mismatch copying ${base}" >&2
        exit 1
    fi
    mv -f "${LIB_DST}/${base}.new" "${LIB_DST}/${base}"
done

# ── Stage .ndf model/data files + inference-engine plugins ────────────────────
# Discovered empirically: FC classification and FMD extraction both fail
# without these, even with a valid license and correctly-staged core libs.
#   - Bin/Data/Fingers.ndf missing -> N_E_FILE_NOT_FOUND (-21),
#     "Fingers classifier model file is missing"
#   - InferenceEngines/ missing -> N_E_ARGUMENT (-10),
#     "None of the specified Inference Engines are available"
# Neither was staged before this fix; both are read-only SDK resources under
# the existing /opt/verifinger mount, so no new volume mount is needed here.
# Recursive + checksum-verified, same principle as the lib staging above.
stage_tree() {
    local src="$1" dst="$2" label="$3"
    echo "[entrypoint] staging ${label} from ${src} -> ${dst}"
    mkdir -p "${dst}"
    while IFS= read -r -d '' f; do
        rel="${f#"${src}"/}"
        dest_file="${dst}/${rel}"
        mkdir -p "$(dirname "${dest_file}")"
        cp -f "$f" "${dest_file}.new"
        src_md5=$(md5sum "$f" | awk '{print $1}')
        dst_md5=$(md5sum "${dest_file}.new" | awk '{print $1}')
        if [ "$src_md5" != "$dst_md5" ]; then
            echo "[entrypoint] FATAL: checksum mismatch copying ${label}/${rel}" >&2
            exit 1
        fi
        mv -f "${dest_file}.new" "${dest_file}"
    done < <(find "${src}" -type f -print0)
}
stage_tree "${SDK_DATA_SRC}" "${DATA_DST}" "SDK data files"
stage_tree "${SDK_INFERENCE_SRC}" "${INFERENCE_DST}" "inference engine plugins"

cp -f "${SDK_PGD_SRC}" "${PGD_DIR}/pgd.new"
src_md5=$(md5sum "${SDK_PGD_SRC}" | awk '{print $1}')
dst_md5=$(md5sum "${PGD_DIR}/pgd.new" | awk '{print $1}')
if [ "$src_md5" != "$dst_md5" ]; then
    echo "[entrypoint] FATAL: checksum mismatch copying pgd binary" >&2
    exit 1
fi
mv -f "${PGD_DIR}/pgd.new" "${PGD_DIR}/pgd"
chmod +x "${PGD_DIR}/pgd"

if [ ! -f "${PGD_CONF}" ]; then
    echo "[entrypoint] writing default pgd.conf from ${PGD_CONF_DEFAULT} (none present in workspace)"
    cp "${PGD_CONF_DEFAULT}" "${PGD_CONF}"
else
    echo "[entrypoint] using existing workspace pgd.conf (not overwriting)"
fi

# ── Start pgd ──────────────────────────────────────────────────────────────────
echo "[entrypoint] starting pgd"
rm -f "${PGD_LOG}"
# pgd daemonizes and returns immediately; `|| true` because a benign non-zero
# exit here must not abort startup under set -e — the readiness checks below
# are the real, authoritative gate, not this launch call's own exit code.
( cd "${PGD_DIR}" && ./pgd ) || true

echo "[entrypoint] waiting for pgd to report started (up to 30s)"
started=0
for i in $(seq 1 30); do
    if grep -q "PG started" "${PGD_LOG}" 2>/dev/null; then
        echo "[entrypoint] pgd confirmed up: $(grep 'PG started' "${PGD_LOG}")"
        started=1
        break
    fi
    sleep 1
done
if [ "${started}" -ne 1 ]; then
    echo "[entrypoint] FATAL: pgd did not report started within 30s" >&2
    cat "${PGD_LOG}" >&2 || echo "[entrypoint] (${PGD_LOG} was never written)" >&2
    exit 1
fi

# ── Verify trial licensing is SERVER-CONFIRMED, not just locally granted ─────
# Learned the hard way: with https=false, pgd reported "PG started" and
# NLicenseObtainA returned available=True from a purely local grant, while the
# real online handshake failed ("(0x8008f201) http: Bad Request (400)") and
# LicenceUsage.log never got a row. "Started" is not "licensed" — actively
# drive a license request and check both the ledger and the error log before
# declaring the container ready.
#
# NOTE: this check does NOT require a *new* LicenceUsage.log row. Tested
# empirically: once a product has been granted once, pgd does not append or
# even touch the ledger on subsequent identical requests (line count and
# mtime both stay unchanged) — StoragePath persists across restarts, so a
# "must gain a new row" check would FATAL on every boot after the first.
# Instead: (a) the ledger must contain a FingerExtractor row with nonzero
# usage — proof it was confirmed at least once, verified reliable/synchronous
# both ways (empty under https=false, populated under https=true, in both
# cases immediately after the NLicenseObtainA call) — and (b) THIS boot's
# freshly-truncated pgd.log must contain no handshake error, as a best-effort
# extra signal.
#
# KNOWN GAP, deliberately accepted: (b) is NOT reliable. Verified by actually
# reintroducing https=false here and re-running this script: the
# 0x8008f201/"Bad Request (400)" error did NOT appear in pgd.log even after
# 90+ seconds of polling — it comes from an async background thread with
# irregular, multi-minute timing (independently confirmed: unrelated
# "ssl failure 1227" lines showed up in a *working* https=true log roughly
# every 8 minutes with no client active at all). So this gate reliably
# catches "this setup has never once been server-confirmed" (the worst
# failure — totally broken from image build), but does NOT reliably catch
# "this used to work but silently broke as of this specific boot" on a later
# restart. Deliberately not fixed by wiping StoragePath every boot to force a
# fresh handshake: trial licenses commonly cap total activation counts, and
# doing that on every restart risks burning through that budget. If ongoing
# health during a long batch run matters, watch /tmp/pgd.log during the run
# rather than trusting this boot-time gate alone.
#
# (a)'s own check is a snapshot read, and got hit by exactly the async delay
# this comment already warned about: confirmed 2026-07-02 that even with
# https=true and a fully correct config, the initial server handshake that
# populates LicenceUsage.log does not always land within the same instant
# NLicenseObtainA returns available=True — a boot FATAL'd here while a
# long-lived manual container with the identical config/libs later showed a
# real, nonzero FingerExtractor row once given more time. So (a) is now a
# poll loop, not a one-shot check, to ride out that lag. 60s is an
# empirically-chosen bound (2x the pgd-startup wait below), not a proven
# worst case — if this still FATALs intermittently, that's a real signal,
# not necessarily a config bug; check pgd.log for handshake activity before
# assuming something is broken.
echo "[entrypoint] verifying server-confirmed trial licensing"

if license_check_out=$(python3 - <<PYEOF
import ctypes, os, sys
LIB_DIR = "${LIB_DST}"
LOAD_ORDER = ["libmkl_tiny.so", "libNCore.so", "libNLicensing.so",
              "libNMedia.so", "libNMediaProc.so", "libNBiometricClient.so"]
libs = {}
for name in LOAD_ORDER:
    libs[name] = ctypes.CDLL(os.path.join(LIB_DIR, name), mode=ctypes.RTLD_GLOBAL)
NCore, NLic = libs["libNCore.so"], libs["libNLicensing.so"]
N_OK = 0
r = NCore.NCoreOnStart()
if r != N_OK:
    print(f"NCoreOnStart failed: {r}")
    sys.exit(1)
r = NLic.NLicManSetTrialMode(True)
if r != N_OK:
    print(f"NLicManSetTrialMode failed: {r}")
    sys.exit(1)
available = ctypes.c_int(0)
r = NLic.NLicenseObtainA(b"/local", b"5000", b"FingerExtractor", ctypes.byref(available))
print(f"NLicenseObtainA result={r} available={bool(available.value)}")
sys.exit(0 if (r == N_OK and available.value) else 1)
PYEOF
); then
    license_check_status=0
else
    license_check_status=$?
fi
echo "[entrypoint] license check: ${license_check_out}"

if grep -qE "0x8008f201|Bad Request \(400\)" "${PGD_LOG}" 2>/dev/null; then
    echo "[entrypoint] FATAL: pgd.log shows the trial/licensing server handshake failed this boot (0x8008f201 / Bad Request 400)." >&2
    echo "[entrypoint] This is the exact https=false failure mode seen before — a local grant without a confirmed license. Check pgd.conf's https= setting and egress to activation1.neurotechnology.com." >&2
    exit 1
fi

if [ "${license_check_status}" -ne 0 ]; then
    echo "[entrypoint] FATAL: NLicenseObtainA did not return available=True this boot (${license_check_out})." >&2
    exit 1
fi

echo "[entrypoint] waiting for licensing server confirmation in ${USAGE_LOG} (up to 60s — this handshake is async, see comment above)"
usage_row=""
for i in $(seq 1 60); do
    usage_row=$(awk -F'\t' '$2=="\"FingerExtractor\""' "${USAGE_LOG}" 2>/dev/null | tail -1)
    if [ -n "${usage_row}" ]; then
        echo "[entrypoint] licensing server confirmation observed after ${i}s"
        break
    fi
    sleep 1
done
if [ -z "${usage_row}" ]; then
    echo "[entrypoint] FATAL: ${USAGE_LOG} has no row for FingerExtractor after 60s — trial licensing has never been server-confirmed." >&2
    exit 1
fi
concurrent_usage=$(echo "${usage_row}" | awk -F'\t' '{print $3}')
nonconcurrent_usage=$(echo "${usage_row}" | awk -F'\t' '{print $5}')
if [ "${concurrent_usage:-0}" = "0" ] && [ "${nonconcurrent_usage:-0}" = "0" ]; then
    echo "[entrypoint] FATAL: ${USAGE_LOG} has a FingerExtractor row but zero usage recorded." >&2
    exit 1
fi

echo "[entrypoint] licensing server-confirmed: ${USAGE_LOG} shows FingerExtractor usage, no handshake errors this boot."

# ── Verify the .ndf data files + inference-engine plugins are FUNCTIONAL ──────
# Same "verify, don't just check presence" principle as the licensing gate
# above: checksum-verified staging only proves the bytes landed correctly,
# not that the SDK can actually load and use them (e.g. a plugin that fails
# to dlopen due to a missing transitive .so dependency would stage fine and
# still break every FC/FMD extraction at use time). Registers the same
# resources the real extraction pipeline needs and confirms at least one
# inference engine plugin actually loads.
echo "[entrypoint] verifying SDK data files + inference engine plugins are functional"

if data_ie_check_out=$(python3 - <<PYEOF
import ctypes, os, sys
LIB_DIR = "${LIB_DST}"
LOAD_ORDER = ["libmkl_tiny.so", "libNCore.so", "libNLicensing.so",
              "libNMedia.so", "libNMediaProc.so", "libNBiometricClient.so",
              "libNBiometrics.so"]
libs = {}
for name in LOAD_ORDER:
    libs[name] = ctypes.CDLL(os.path.join(LIB_DIR, name), mode=ctypes.RTLD_GLOBAL)
NCore, NBiom = libs["libNCore.so"], libs["libNBiometrics.so"]
N_OK = 0
c_void_p = ctypes.c_void_p

r = NCore.NCoreOnStart()
if r != N_OK:
    print(f"NCoreOnStart failed: {r}")
    sys.exit(1)

hDataDir = c_void_p()
r = NCore.NStringCreateFromStrOrCharsA(b"${DATA_DST}", -1, ctypes.byref(hDataDir))
if r != N_OK:
    print(f"NStringCreateFromStrOrCharsA failed: {r}")
    sys.exit(1)
hDataMgr = c_void_p()
r = NCore.NDataFileManagerGetInstance(ctypes.byref(hDataMgr))
if r != N_OK:
    print(f"NDataFileManagerGetInstance failed: {r}")
    sys.exit(1)
r = NCore.NDataFileManagerAddFromDirectoryN(hDataMgr, hDataDir, 1)
if r != N_OK:
    print(f"NDataFileManagerAddFromDirectoryN failed: {r}")
    sys.exit(1)

hIeMgr = c_void_p()
r = NBiom.NInferenceEngineManagerGetPluginManager(ctypes.byref(hIeMgr))
if r != N_OK:
    print(f"NInferenceEngineManagerGetPluginManager failed: {r}")
    sys.exit(1)
r = NBiom.NPluginManagerSetPluginSearchPathA(hIeMgr, b"${INFERENCE_DST}")
if r != N_OK:
    print(f"NPluginManagerSetPluginSearchPathA failed: {r}")
    sys.exit(1)
r = NBiom.NPluginManagerPlugAll(hIeMgr)
if r != N_OK:
    print(f"NPluginManagerPlugAll failed: {r}")
    sys.exit(1)
pluginCount = ctypes.c_int(0)
r = NBiom.NPluginManagerGetPluginCount(hIeMgr, ctypes.byref(pluginCount))
if r != N_OK:
    print(f"NPluginManagerGetPluginCount failed: {r}")
    sys.exit(1)
print(f"data files registered ok, inference engine plugins loaded: {pluginCount.value}")
sys.exit(0 if pluginCount.value > 0 else 1)
PYEOF
); then
    data_ie_check_status=0
else
    data_ie_check_status=$?
fi
echo "[entrypoint] data/inference engine check: ${data_ie_check_out}"

if [ "${data_ie_check_status}" -ne 0 ]; then
    echo "[entrypoint] FATAL: SDK data files or inference engine plugins failed to load functionally (${data_ie_check_out})." >&2
    echo "[entrypoint] Files staged with correct checksums but the SDK could not use them — check for missing transitive .so dependencies or permission issues under ${INFERENCE_DST}." >&2
    exit 1
fi

echo "[entrypoint] container ready, sleeping"
exec sleep infinity

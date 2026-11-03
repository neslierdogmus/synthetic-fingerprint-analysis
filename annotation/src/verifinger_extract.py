"""Shared VeriFinger ctypes bindings: FC (pattern classification) and FMD
(minutiae) extraction from a fingerprint image.

Call init() once per process before classify()/extract_minutiae(). Not
thread-safe; run one process per job (see CLAUDE.md's tmux-per-job
convention for anything beyond a single-process --dev smoke test).

Gotchas this module bakes in (see /workspace/CONTEXT.md for the empirical
findings behind each one -- do not "simplify" these away):
  - NFloat (image resolution) is a 32-bit C float, not a double. Passing a
    c_double silently corrupts the SysV x86-64 calling convention and every
    image is rejected with nbsInvalidSampleResolution (905).
  - Real-world source images here carry no/bogus DPI tags; VeriFinger
    requires a resolution to be set explicitly before use, so this module
    always forces a caller-supplied DPI (default 500, the true spec for
    every dataset used so far).
  - Any non-Grayscale8U pixel format (RGB, or the indexed-palette encoding
    some .bmp files load as) is converted to Grayscale8U before use --
    handles both the RGB and BMP-indexed cases with one generic path.
  - NSubjectGetFinger(subject, 1, ...) is intentionally 1-based (index 0 is
    the input finger; 1 is the finger after task processing).
  - The `NFMinutia.Angle` raw byte needs BOTH: RotationToDegrees
    ((2*r*360+256)//(2*256)) to get VeriFinger's own degrees, AND a winding
    inversion ((360-deg) % 360) to match this project's ground-truth
    convention (FVC_BERKAY_TOPCU .npz). Skipping either one silently
    produces wrong angles -- confirmed via nearest-neighbor accuracy
    against real ground truth, not just by reading the SDK docs.
  - `Fingers.DeterminePatternClass`/`Fingers.TemplateSize` TypeOf function
    pointers live in different .so files (NBooleanTypeOf in libNCore.so,
    NTemplateSizeTypeOf in libNBiometrics.so) -- get the wrong one and
    ctypes raises AttributeError, not a VeriFinger error.
"""
import ctypes
import os

LIB_DIR = "/workspace/annotation_src/lib_linux_x86_64"
DATA_DIR = "/workspace/annotation_src/data"
PLUGIN_DIR = os.path.join(LIB_DIR, "InferenceEngines")

_LOAD_ORDER = [
    "libmkl_tiny.so",
    "libNCore.so",
    "libNLicensing.so",
    "libNMedia.so",
    "libNMediaProc.so",
    "libNBiometrics.so",
    "libNBiometricClient.so",
]

N_OK = 0
NBS_OK = 1
NBS_OPERATION_NOT_ACTIVATED = 991  # license dropped mid-task; see reobtain_license()
NPF_GRAYSCALE_8U = 0x00301001
NFP_UNKNOWN = 0
NBO_DETECT_SEGMENTS = 4
NTS_LARGE = 256

# NFPatternClass (NBiometricTypes.h) -> spec's {A,L,R,T,W,S}. nfpUnknown
# position means VeriFinger only ever returns the slant pair (10/11), never
# the radial/ulnar pair (3/4) -- confirmed empirically, see CONTEXT.md.
PATTERN_CLASS_TO_LETTER = {
    0: None,   # nfpcUnknown
    1: "A",    # nfpcPlainArch
    2: "T",    # nfpcTentedArch
    3: None,   # nfpcRadialLoop -- not expected with position=nfpUnknown
    4: None,   # nfpcUlnarLoop -- not expected with position=nfpUnknown
    5: "W",    # nfpcPlainWhorl
    6: "W",    # nfpcCentralPocketLoop
    7: "W",    # nfpcDoubleLoop
    8: "W",    # nfpcAccidentalWhorl
    9: "W",    # nfpcWhorl
    10: "R",   # nfpcRightSlantLoop
    11: "L",   # nfpcLeftSlantLoop
    12: "S",   # nfpcScar
    15: None,  # nfpcAmputation
}


class NFMinutia(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_ushort),
        ("Y", ctypes.c_ushort),
        ("Type", ctypes.c_int),
        ("Angle", ctypes.c_ubyte),
        ("Quality", ctypes.c_ubyte),
        ("Curvature", ctypes.c_ubyte),
        ("G", ctypes.c_ubyte),
    ]


_libs = {}
_inited = False


def _check(result, msg=""):
    if result != N_OK:
        raise RuntimeError(f"VeriFinger call failed (result={result}) {msg}")


def init():
    global _inited
    if _inited:
        return
    for name in _LOAD_ORDER:
        _libs[name] = ctypes.CDLL(os.path.join(LIB_DIR, name), mode=ctypes.RTLD_GLOBAL)
    core = _libs["libNCore.so"]
    lic = _libs["libNLicensing.so"]
    bio = _libs["libNBiometrics.so"]

    _check(core.NCoreOnStart())
    _check(lic.NLicManSetTrialMode(True))

    available = ctypes.c_int(0)
    _check(lic.NLicenseObtainA(b"/local", b"5000", b"FingerClient,FingerExtractor", ctypes.byref(available)))
    if not available.value:
        raise RuntimeError("VeriFinger license not available (NLicenseObtainA available=False)")

    mgr = ctypes.c_void_p()
    _check(core.NDataFileManagerGetInstance(ctypes.byref(mgr)))
    hstr = ctypes.c_void_p()
    _check(core.NStringCreateFromStrOrCharsA(DATA_DIR.encode(), -1, ctypes.byref(hstr)))
    _check(core.NDataFileManagerAddFromDirectoryN(mgr, hstr, True))
    _release(hstr)

    pluginmgr = ctypes.c_void_p()
    _check(bio.NInferenceEngineManagerGetPluginManager(ctypes.byref(pluginmgr)))
    _check(core.NPluginManagerSetPluginSearchPathA(pluginmgr, PLUGIN_DIR.encode()))
    _check(core.NPluginManagerPlugAll(pluginmgr))

    _inited = True


def reobtain_license():
    """Re-runs NLicenseObtainA. The pgd<->licensing-server connection can
    drop and auto-restore mid-run (observed: dropped and restored 4 minutes
    apart during a real batch run, see CONTEXT.md) without the trial grant
    in this process ever refreshing itself -- every classify()/
    extract_minutiae() call kept failing for the rest of that run even
    after /tmp/pgd.log logged the connection as restored. Called
    automatically by classify()/extract_minutiae() on a license-shaped
    failure; safe to call anytime."""
    lic = _libs["libNLicensing.so"]
    available = ctypes.c_int(0)
    lic.NLicenseObtainA(b"/local", b"5000", b"FingerClient,FingerExtractor", ctypes.byref(available))
    return bool(available.value)


def _release(handle):
    if handle.value:
        _libs["libNCore.so"].NObjectSet(None, ctypes.byref(handle))


def _load_grayscale_image(path, dpi):
    """Decodes via PIL and hands VeriFinger raw grayscale pixel bytes
    directly, rather than NImageCreateFromFileExA + NImageSetHorzResolution
    on VeriFinger's own file decode. Necessary, not just a style choice:
    JPEG files (.jpg/.jpeg/.JPEG, 5 of the 11 /data/synthetic sources)
    carry a JFIF density marker with unit=0 ("aspect ratio only, not a
    real resolution"), and VeriFinger's native JPEG decoder appears to
    latch an internal "no valid resolution" flag from that marker that our
    later NImageSetHorzResolution/VertResolution override doesn't fully
    clear -- every JPEG failed with nbsInvalidSampleResolution (905) even
    though NImageGetHorzResolution read back the overridden value
    correctly. TIFF/BMP/PNG sources don't carry that marker and were
    unaffected, which is what made this JPEG-specific bug easy to miss
    initially. Loading via PIL sidesteps VeriFinger's file-format parsing
    (and whatever it infers from format-specific metadata) entirely, and
    PIL's .convert("L") also subsumes the old NImageCreateFromImageEx2
    RGB/indexed-palette-to-grayscale conversion for free."""
    from PIL import Image
    bio = _libs["libNBiometrics.so"]
    img = Image.open(path).convert("L")
    w, h = img.size
    data = img.tobytes()  # tight-packed, stride == width for mode "L"
    hImage = ctypes.c_void_p()
    _check(bio.NImageCreateFromDataEx(NPF_GRAYSCALE_8U, w, h, w, w, data, len(data), 0, ctypes.byref(hImage)), path)
    _check(bio.NImageSetHorzResolution(hImage, ctypes.c_float(dpi)))
    _check(bio.NImageSetVertResolution(hImage, ctypes.c_float(dpi)))
    return hImage


def _make_finger_subject(hImage):
    bio = _libs["libNBiometrics.so"]
    hFinger = ctypes.c_void_p()
    _check(bio.NFingerCreate(ctypes.byref(hFinger)))
    _check(bio.NFrictionRidgeSetImage(hFinger, hImage))
    _check(bio.NFrictionRidgeSetPosition(hFinger, NFP_UNKNOWN))
    hSubject = ctypes.c_void_p()
    _check(bio.NSubjectCreate(ctypes.byref(hSubject)))
    _check(bio.NSubjectAddFinger(hSubject, hFinger, None))
    return hFinger, hSubject


def classify(path, dpi=500.0, _retried=False):
    """Returns dict(letter, raw_class, confidence, status) for one image.
    letter is None if VeriFinger returned a class with no {A,L,R,T,W,S}
    mapping (nfpcUnknown/nfpcAmputation/radial/ulnar) or the task failed.
    On NBS_OPERATION_NOT_ACTIVATED (a mid-run license drop, see
    reobtain_license()) retries the whole call once after re-obtaining."""
    result = _classify_impl(path, dpi)
    if result["status"] == NBS_OPERATION_NOT_ACTIVATED and not _retried:
        reobtain_license()
        return classify(path, dpi, _retried=True)
    return result


def _classify_impl(path, dpi):
    core, bio, cli = _libs["libNCore.so"], _libs["libNBiometrics.so"], _libs["libNBiometricClient.so"]
    hImage = _load_grayscale_image(path, dpi)
    hFinger, hSubject = _make_finger_subject(hImage)

    hClient = ctypes.c_void_p()
    _check(cli.NBiometricClientCreate(ctypes.byref(hClient)))
    param = ctypes.c_int(1)  # NTrue
    _check(core.NObjectSetPropertyPA(hClient, b"Fingers.DeterminePatternClass", core.NBooleanTypeOf,
                                      0, ctypes.byref(param), ctypes.sizeof(param), 1, 1))
    hTask = ctypes.c_void_p()
    _check(bio.NBiometricEngineCreateTask(hClient, NBO_DETECT_SEGMENTS, hSubject, None, ctypes.byref(hTask)))
    _check(bio.NBiometricEnginePerformTask(hClient, hTask))
    status = ctypes.c_int(0)
    _check(bio.NBiometricTaskGetStatus(hTask, ctypes.byref(status)))

    letter, raw, conf = None, None, None
    if status.value == NBS_OK:
        hFinger2 = ctypes.c_void_p()
        _check(bio.NSubjectGetFinger(hSubject, 1, ctypes.byref(hFinger2)))
        hAttrs = ctypes.c_void_p()
        _check(bio.NFrictionRidgeGetObject(hFinger2, 0, ctypes.byref(hAttrs)))
        patternClass = ctypes.c_int(0)
        _check(bio.NFAttributesGetPatternClass(hAttrs, ctypes.byref(patternClass)))
        confidence = ctypes.c_ubyte(0)
        _check(bio.NFAttributesGetPatternClassConfidence(hAttrs, ctypes.byref(confidence)))
        raw = patternClass.value
        conf = confidence.value
        letter = PATTERN_CLASS_TO_LETTER.get(raw)
        _release(hAttrs)
        _release(hFinger2)

    _release(hTask)
    _release(hClient)
    _release(hSubject)
    _release(hFinger)
    _release(hImage)
    return {"letter": letter, "raw_class": raw, "confidence": conf, "status": status.value}


def extract_minutiae(path, dpi=500.0, _retried=False):
    """Returns (minutiae, status) where minutiae is a list of
    (x, y, type, angle_deg) tuples. type: 1=end, 2=bifurcation (matches
    LABEL_FORMAT.md); angle_deg has both documented conversions applied.
    Unlike classify()'s task-status path, a mid-run license drop makes
    NBiometricEngineCreateTemplate itself return a failing NResult
    (N_E_AGGREGATE / -33), which _check() turns into a RuntimeError here
    -- caught and retried once after reobtain_license(), same trigger as
    classify()'s NBS_OPERATION_NOT_ACTIVATED path."""
    try:
        return _extract_minutiae_impl(path, dpi)
    except RuntimeError:
        if _retried:
            raise
        reobtain_license()
        return extract_minutiae(path, dpi, _retried=True)


def _extract_minutiae_impl(path, dpi):
    core, bio, cli = _libs["libNCore.so"], _libs["libNBiometrics.so"], _libs["libNBiometricClient.so"]
    hImage = _load_grayscale_image(path, dpi)
    hFinger, hSubject = _make_finger_subject(hImage)

    hClient = ctypes.c_void_p()
    _check(cli.NBiometricClientCreate(ctypes.byref(hClient)))
    param = ctypes.c_int(NTS_LARGE)
    _check(core.NObjectSetPropertyPA(hClient, b"Fingers.TemplateSize", bio.NTemplateSizeTypeOf,
                                      0, ctypes.byref(param), ctypes.sizeof(param), 1, 1))
    status = ctypes.c_int(0)
    _check(bio.NBiometricEngineCreateTemplate(hClient, hSubject, ctypes.byref(status)))

    minutiae = []
    if status.value == NBS_OK:
        hTemplate = ctypes.c_void_p()
        _check(bio.NSubjectGetTemplate(hSubject, ctypes.byref(hTemplate)))
        hFTemplate = ctypes.c_void_p()
        _check(bio.NTemplateGetFingersEx(hTemplate, ctypes.byref(hFTemplate)))
        hRecord = ctypes.c_void_p()
        _check(bio.NFTemplateGetRecordEx(hFTemplate, 0, ctypes.byref(hRecord)))
        count = ctypes.c_int(0)
        _check(bio.NFRecordGetMinutiaCount(hRecord, ctypes.byref(count)))
        for i in range(count.value):
            m = NFMinutia()
            _check(bio.NFRecordGetMinutia(hRecord, i, ctypes.byref(m)))
            deg_raw = (2 * m.Angle * 360 + 256) // (2 * 256)
            deg = (360 - deg_raw) % 360
            minutiae.append((int(m.X), int(m.Y), int(m.Type), deg))
        _release(hRecord)
        _release(hFTemplate)
        _release(hTemplate)

    _release(hClient)
    _release(hSubject)
    _release(hFinger)
    _release(hImage)
    return minutiae, status.value

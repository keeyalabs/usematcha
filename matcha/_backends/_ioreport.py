# SPDX-License-Identifier: Apache-2.0
"""matcha._backends._ioreport — ctypes binding to Apple's private IOReport.

Apple doesn't ship an NVML-equivalent, but the Darwin kernel exposes a
private framework — ``IOReport`` — that has done the job for more than
a decade: it streams cumulative energy counters per SoC subsystem
(GPU, GPU SRAM, ANE, CPU cores, DRAM, …) in millijoules, with no root
requirement. This is the same API ``powermetrics``, ``asitop``,
``macmon``, and iStat Menus all go through to avoid sudo.

What this module is
-------------------

A thin ctypes binding exposing exactly what matcha needs — **not** a
general-purpose IOReport wrapper. We enumerate the ``"Energy Model"``
group, filter to GPU channels (``"GPU Energy"``, ``"GPU SRAM"``), and
expose an object that, once opened, can be polled for the cumulative
GPU energy consumed since ``open()``.

No pip dependency: ctypes is stdlib, the IOReport framework ships with
every Mac, the CoreFoundation framework ships with every Mac. Pure
syscalls-over-ctypes, no binary wheel.

Private-API caveat
------------------

IOReport is not part of Apple's public SDK. Symbol names have been
stable across macOS 10.15 → 15.x, but Apple could, in principle,
break them. If ``_load_symbols()`` fails we surface a clear error and
``AppleSiliconBackend`` reports unavailable — the CLI auto-detect
falls through cleanly and matcha never crashes because the private API
moved.

References
----------

* Zeus's ``zeus-apple-silicon`` C++ header (Apache 2.0) — reference
  for function signatures, channel group names, and unit conversion
  table. matcha's implementation is independent, in Python, and
  limited to the GPU subset we need.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import threading
from ctypes import c_int, c_int32, c_int64, c_uint32, c_uint64, c_void_p, c_char_p
from typing import Dict, List, Optional


# ---- framework loading ---------------------------------------------------
#
# macOS moved private frameworks out of on-disk ``.framework`` bundles
# and into the dyld shared cache in Big Sur (11.0). IOReport in particular
# ships today as ``/usr/lib/libIOReport.dylib`` on all supported macOS
# versions (this is what ``/usr/bin/powermetrics`` itself links against;
# ``otool -L /usr/bin/powermetrics`` confirms it). On older macOS it also
# existed at ``/System/Library/PrivateFrameworks/IOReport.framework/IOReport``
# — we try that as a fallback for the long tail.
_IOREPORT_PATHS = (
    "/usr/lib/libIOReport.dylib",
    "/System/Library/PrivateFrameworks/IOReport.framework/IOReport",
    "/System/Library/PrivateFrameworks/IOReport.framework/Versions/A/IOReport",
)
_COREFOUNDATION_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"

_kCFStringEncodingUTF8 = 0x08000100


class IOReportUnavailable(RuntimeError):
    """Raised when IOReport can't be loaded or bound on this host."""


def _try_load(path: str) -> Optional[ctypes.CDLL]:
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


def _load_ioreport() -> ctypes.CDLL:
    for p in _IOREPORT_PATHS:
        lib = _try_load(p)
        if lib is not None:
            return lib
    raise IOReportUnavailable(
        "cannot load IOReport — tried " + ", ".join(_IOREPORT_PATHS)
    )


def _load_symbols():
    """Load IOReport + CoreFoundation and bind the functions we use.

    Returns a namespace object with every needed function pre-typed.
    Raises IOReportUnavailable if either library or any required
    symbol is missing.
    """
    ioreport = _load_ioreport()
    cf = _try_load(_COREFOUNDATION_PATH)
    if cf is None:
        raise IOReportUnavailable(
            f"cannot load CoreFoundation framework at {_COREFOUNDATION_PATH}"
        )

    class _Syms:
        pass

    s = _Syms()

    # ---- IOReport --------------------------------------------------------
    # Signatures are based on `zeus-apple-silicon/apple_energy/apple_energy.hpp`
    # (Apache 2.0), cross-checked against a handful of IOReport headers
    # that have leaked into open-source ecosystems (macmon, powermetrics
    # disassembly). The "unused" parameters are genuinely ignored by
    # the framework — we pass NULL / 0 for them.

    s.IOReportCopyChannelsInGroup = ioreport.IOReportCopyChannelsInGroup
    s.IOReportCopyChannelsInGroup.argtypes = [c_void_p, c_void_p, c_uint64, c_uint64, c_uint64]
    s.IOReportCopyChannelsInGroup.restype = c_void_p  # CFDictionaryRef

    s.IOReportCreateSubscription = ioreport.IOReportCreateSubscription
    s.IOReportCreateSubscription.argtypes = [c_void_p, c_void_p, c_void_p, c_uint64, c_void_p]
    s.IOReportCreateSubscription.restype = c_void_p

    s.IOReportCreateSamples = ioreport.IOReportCreateSamples
    s.IOReportCreateSamples.argtypes = [c_void_p, c_void_p, c_void_p]
    s.IOReportCreateSamples.restype = c_void_p  # CFDictionaryRef

    s.IOReportCreateSamplesDelta = ioreport.IOReportCreateSamplesDelta
    s.IOReportCreateSamplesDelta.argtypes = [c_void_p, c_void_p, c_void_p]
    s.IOReportCreateSamplesDelta.restype = c_void_p  # CFDictionaryRef

    s.IOReportSimpleGetIntegerValue = ioreport.IOReportSimpleGetIntegerValue
    s.IOReportSimpleGetIntegerValue.argtypes = [c_void_p, c_int]
    s.IOReportSimpleGetIntegerValue.restype = c_int64

    s.IOReportChannelGetChannelName = ioreport.IOReportChannelGetChannelName
    s.IOReportChannelGetChannelName.argtypes = [c_void_p]
    s.IOReportChannelGetChannelName.restype = c_void_p  # CFStringRef (borrowed)

    s.IOReportChannelGetUnitLabel = ioreport.IOReportChannelGetUnitLabel
    s.IOReportChannelGetUnitLabel.argtypes = [c_void_p]
    s.IOReportChannelGetUnitLabel.restype = c_void_p  # CFStringRef (borrowed)

    # ---- CoreFoundation --------------------------------------------------

    s.CFRelease = cf.CFRelease
    s.CFRelease.argtypes = [c_void_p]
    s.CFRelease.restype = None

    s.CFStringCreateWithCString = cf.CFStringCreateWithCString
    s.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
    s.CFStringCreateWithCString.restype = c_void_p

    s.CFStringGetCStringPtr = cf.CFStringGetCStringPtr
    s.CFStringGetCStringPtr.argtypes = [c_void_p, c_uint32]
    s.CFStringGetCStringPtr.restype = c_char_p

    s.CFStringGetCString = cf.CFStringGetCString
    s.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_int64, c_uint32]
    s.CFStringGetCString.restype = c_int  # Boolean

    s.CFDictionaryCreateMutableCopy = cf.CFDictionaryCreateMutableCopy
    s.CFDictionaryCreateMutableCopy.argtypes = [c_void_p, c_int64, c_void_p]
    s.CFDictionaryCreateMutableCopy.restype = c_void_p

    s.CFDictionaryGetCount = cf.CFDictionaryGetCount
    s.CFDictionaryGetCount.argtypes = [c_void_p]
    s.CFDictionaryGetCount.restype = c_int64

    s.CFDictionaryGetValue = cf.CFDictionaryGetValue
    s.CFDictionaryGetValue.argtypes = [c_void_p, c_void_p]
    s.CFDictionaryGetValue.restype = c_void_p  # borrowed

    s.CFArrayGetCount = cf.CFArrayGetCount
    s.CFArrayGetCount.argtypes = [c_void_p]
    s.CFArrayGetCount.restype = c_int64

    s.CFArrayGetValueAtIndex = cf.CFArrayGetValueAtIndex
    s.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_int64]
    s.CFArrayGetValueAtIndex.restype = c_void_p  # borrowed

    return s


_syms: Optional[object] = None
_load_err: Optional[str] = None
_load_lock = threading.Lock()


def _symbols():
    """Lazy-load symbols on first use, cache on success.

    Deferring the load keeps ``is_available()`` checks cheap on
    non-Apple-Silicon hosts (where auto-detect skips this backend
    anyway) and means an ``import matcha`` on macOS never dlopens a
    private framework just for existing.
    """
    global _syms, _load_err
    with _load_lock:
        if _syms is not None:
            return _syms
        if _load_err is not None:
            raise IOReportUnavailable(_load_err)
        try:
            _syms = _load_symbols()
            return _syms
        except IOReportUnavailable as e:
            _load_err = str(e)
            raise


# ---- CFString helpers ----------------------------------------------------


def _cf_string_from(py_str: str) -> c_void_p:
    """Create a retained CFStringRef. Caller must CFRelease."""
    s = _symbols()
    ptr = s.CFStringCreateWithCString(
        None, py_str.encode("utf-8"), _kCFStringEncodingUTF8
    )
    if not ptr:
        raise IOReportUnavailable(f"CFStringCreateWithCString failed for {py_str!r}")
    return c_void_p(ptr)


def _cf_string_to_py(cfstr: Optional[int]) -> str:
    """Read a (borrowed) CFStringRef into a Python string.

    Uses the fast ``CFStringGetCStringPtr`` path when CF has a direct
    UTF-8 buffer, falls back to ``CFStringGetCString`` into a local
    buffer otherwise. Returns ``""`` on a null or unreadable string.
    """
    if not cfstr:
        return ""
    s = _symbols()
    fast = s.CFStringGetCStringPtr(cfstr, _kCFStringEncodingUTF8)
    if fast:
        return ctypes.string_at(fast).decode("utf-8", errors="replace")
    buf = ctypes.create_string_buffer(512)
    if s.CFStringGetCString(cfstr, buf, len(buf), _kCFStringEncodingUTF8):
        return buf.value.decode("utf-8", errors="replace")
    return ""


# ---- unit conversion -----------------------------------------------------

# IOReport channels report integer counts in a per-channel unit, not a
# fixed one. ``IOReportChannelGetUnitLabel`` returns strings like
# "nJ", "µJ", "mJ", "J". We normalize everything to millijoules so the
# backend can expose a single cumulative mJ counter. Table is derived
# from Zeus's ``convert_to_mj`` — the SI prefixes are standard but IOReport
# uses a couple of archaic units (cJ, daJ, hJ) we accept for safety.
_UNIT_TO_MJ = {
    "pj": 1e-9,
    "nj": 1e-6,
    "uj": 1e-3,
    "µj": 1e-3,
    "mj": 1.0,
    "cj": 10.0,
    "dj": 100.0,
    "j": 1000.0,
    "daj": 1e4,
    "hj": 1e5,
    "kj": 1e6,
    "mj_mega": 1e9,   # "MJ" — disambiguated below
}


def _to_mj(value: int, unit: str) -> float:
    """Convert a raw IOReport integer sample to millijoules."""
    u = unit.strip().lower()
    # Disambiguate "mJ" (milli) vs "MJ" (mega). Case matters in the
    # label but we've already lowercased; fall back to inspecting the
    # original string if we get a bare "mj" that's actually mega.
    if unit.strip() == "MJ":
        return float(value) * 1e9
    factor = _UNIT_TO_MJ.get(u)
    if factor is None:
        # Unknown unit → return zero rather than guess. Keeps bad
        # samples out of cumulative counters.
        return 0.0
    return float(value) * factor


# ---- channel classification ---------------------------------------------

# IOReport channel names follow a small set of conventions. We only
# care about GPU-related entries. The main SoC has a channel literally
# named "GPU Energy"; the SRAM cache shows up as "GPU SRAM" on M-series
# with a dedicated SRAM counter. Ultra-class chips prefix channels with
# a die id (e.g. "DIE_0_GPU Energy") — we strip the prefix before
# matching so Ultra machines sum both dies cleanly.
_DIE_PREFIX_RE = re.compile(r"^DIE_\d+_")


def _is_gpu_channel(name: str) -> bool:
    """Return True iff this Energy-Model channel is a GPU counter.

    We match GPU package energy and GPU SRAM; both go into the single
    GPU mJ bucket matcha exposes. Callers that want them separated can
    use ``classify_gpu_channel`` instead.
    """
    n = _DIE_PREFIX_RE.sub("", name).strip()
    nl = n.lower()
    return nl == "gpu energy" or nl.startswith("gpu sram")


# ---- session -------------------------------------------------------------


class IOReportSession:
    """Cumulative GPU energy counter over IOReport's Energy Model group.

    Usage::

        sess = IOReportSession()
        sess.open()
        # ... later, periodically ...
        mj = sess.energy_delta_mj()   # mJ consumed since the previous call
        total = sess.cumulative_mj    # mJ consumed since open()
        sess.close()

    Thread-safety: ``energy_delta_mj`` takes an internal lock and is
    safe to call from one refresher thread while another thread reads
    ``cumulative_mj``.

    Memory management: every CF object we retain is released in
    ``close()`` or in the context manager's ``__exit__``. Borrowed
    references from Get-rule calls are never released.
    """

    def __init__(self) -> None:
        self._sub: Optional[c_void_p] = None
        self._channels_mut: Optional[c_void_p] = None
        self._prev_sample: Optional[c_void_p] = None
        self._io_channels_key: Optional[c_void_p] = None

        self._lock = threading.Lock()
        self._cumulative_mj: float = 0.0
        self._opened = False

    # ---- lifecycle ------------------------------------------------------

    def open(self) -> None:
        s = _symbols()

        group_key = _cf_string_from("Energy Model")
        channels = c_void_p(s.IOReportCopyChannelsInGroup(group_key, None, 0, 0, 0))
        s.CFRelease(group_key)
        if not channels:
            raise IOReportUnavailable(
                "IOReportCopyChannelsInGroup('Energy Model') returned NULL"
            )

        channels_mut = c_void_p(s.CFDictionaryCreateMutableCopy(None, 0, channels))
        s.CFRelease(channels)
        if not channels_mut:
            raise IOReportUnavailable("CFDictionaryCreateMutableCopy failed")

        # Subscription's out-param for updated channels is documented as
        # ignorable; we pass a dummy pointer so the framework has
        # somewhere to scribble if it wants to.
        updated_out = c_void_p()
        sub = c_void_p(s.IOReportCreateSubscription(
            None, channels_mut, ctypes.byref(updated_out), 0, None
        ))
        if not sub:
            s.CFRelease(channels_mut)
            raise IOReportUnavailable("IOReportCreateSubscription returned NULL")

        prev = c_void_p(s.IOReportCreateSamples(sub, channels_mut, None))
        if not prev:
            s.CFRelease(sub)
            s.CFRelease(channels_mut)
            raise IOReportUnavailable("IOReportCreateSamples (initial) returned NULL")

        # Cache the "IOReportChannels" key once — it's used on every
        # delta-iteration call, and CFString creation is not free.
        self._io_channels_key = _cf_string_from("IOReportChannels")

        self._sub = sub
        self._channels_mut = channels_mut
        self._prev_sample = prev
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        s = _symbols()
        if self._prev_sample:
            s.CFRelease(self._prev_sample)
            self._prev_sample = None
        if self._sub:
            s.CFRelease(self._sub)
            self._sub = None
        if self._channels_mut:
            s.CFRelease(self._channels_mut)
            self._channels_mut = None
        if self._io_channels_key:
            s.CFRelease(self._io_channels_key)
            self._io_channels_key = None
        self._opened = False

    # Context-manager sugar so ad-hoc callers can write `with IOReportSession() as io:`
    # without a try/finally. The backend itself uses explicit open()/close().
    def __enter__(self) -> "IOReportSession":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- sampling ------------------------------------------------------

    def energy_delta_mj(self) -> float:
        """Capture a new sample, return GPU mJ consumed since the previous call.

        Also updates ``cumulative_mj``. Thread-safe — safe to call
        concurrently from the refresher thread and the engine's
        step-boundary path (we need both: the refresher keeps power
        current, the boundary call forces a counter-exact read).

        The lock wraps the full critical section (CF roundtrips +
        baseline-sample rotation + cumulative update). Narrower locks
        would race the ``CFRelease(self._prev_sample) ; self._prev_sample
        = curr`` rotation and double-free the baseline.
        """
        if not self._opened:
            raise RuntimeError("IOReportSession.energy_delta_mj called before open()")

        s = _symbols()
        with self._lock:
            curr = c_void_p(s.IOReportCreateSamples(self._sub, self._channels_mut, None))
            if not curr:
                return 0.0

            delta = c_void_p(s.IOReportCreateSamplesDelta(self._prev_sample, curr, None))
            if not delta:
                s.CFRelease(curr)
                return 0.0

            gpu_mj = 0.0
            try:
                channels_array = c_void_p(
                    s.CFDictionaryGetValue(delta, self._io_channels_key)
                )
                if channels_array:
                    n = s.CFArrayGetCount(channels_array)
                    for i in range(n):
                        item = c_void_p(s.CFArrayGetValueAtIndex(channels_array, i))
                        if not item:
                            continue
                        name = _cf_string_to_py(s.IOReportChannelGetChannelName(item))
                        if not _is_gpu_channel(name):
                            continue
                        unit = _cf_string_to_py(s.IOReportChannelGetUnitLabel(item))
                        raw = s.IOReportSimpleGetIntegerValue(item, 0)
                        # Deltas can legally be 0 on idle sub-windows. Negative
                        # values would indicate a counter wrap or a clock
                        # anomaly — drop them to keep cumulative monotonic.
                        if raw <= 0:
                            continue
                        gpu_mj += _to_mj(raw, unit)
            finally:
                s.CFRelease(delta)
                # Rotate the baseline sample — the old prev is released,
                # the new current becomes prev for the next call.
                s.CFRelease(self._prev_sample)
                self._prev_sample = curr

            self._cumulative_mj += gpu_mj
            return gpu_mj

    # ---- accessors -----------------------------------------------------

    @property
    def cumulative_mj(self) -> float:
        with self._lock:
            return self._cumulative_mj


# ---- availability probe --------------------------------------------------


def is_supported() -> bool:
    """Cheap check: can we open IOReport on this host at all?

    Used by the Apple backend's ``is_available()`` — intentionally
    side-effect-free beyond dlopen. No subscription is created, so no
    kernel-level resources are held after this returns.
    """
    try:
        _symbols()
        return True
    except IOReportUnavailable:
        return False

"""Stop the import on x86_64 CPUs that are older than x86-64-v3.

The x86_64 wheels are compiled for x86-64-v3 (AVX2, FMA, BMI2: Intel Haswell 2013 or later, AMD
Zen 2017 or later). On an older CPU, the extension stops the interpreter with an illegal
instruction. To prevent this, this check runs first and raises ImportError. ARM builds do not
need the check, because FMA and NEON are part of their baseline.

To disable the check, set MANYFACED_SKIP_CPU_CHECK=1. This is necessary, for example, in an
emulator that has AVX2 but does not show it in CPUID.

On Windows on Arm, a native arm64 Python gets the arm64 wheel and no check. An x64 Python gets
the x86_64 wheel and runs it in emulation (Prism). The check then sees the features that the
emulator shows. If the emulator does not show AVX2, the import gives ImportError, not a crash.
"""

from __future__ import annotations

import os
import platform
import sys

_REQUIRED = ("avx2", "fma", "bmi2")


def _x86_features() -> set[str] | None:
    """Give the required features that the CPU shows, or None if they are not known."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("flags"):
                        return set(line.split(":", 1)[1].split()) & set(_REQUIRED)
        except OSError:
            return None
        return None
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None)
        found = set()
        for feature, key in (
            ("avx2", b"hw.optional.avx2_0"),
            ("fma", b"hw.optional.fma"),
            ("bmi2", b"hw.optional.bmi2"),
        ):
            value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
            rc = libc.sysctlbyname(key, ctypes.byref(value), ctypes.byref(size), None, 0)
            if rc == 0 and value.value == 1:
                found.add(feature)
        return found
    if sys.platform == "win32":
        import ctypes

        # PF_AVX2_INSTRUCTIONS_AVAILABLE. Each CPU with AVX2 also has FMA3 and BMI2.
        if ctypes.windll.kernel32.IsProcessorFeaturePresent(40):
            return set(_REQUIRED)
        # Windows builds before Windows 11 (22000) can say "no" because they do not know this
        # feature flag. In that case, the features are not known.
        if sys.getwindowsversion().build < 22000:
            return None
        return set()
    return None


def check() -> None:
    """Raise ImportError if the CPU does not have the x86-64-v3 features."""
    if os.environ.get("MANYFACED_SKIP_CPU_CHECK") == "1":
        return
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return
    features = _x86_features()
    if features is None:
        # The platform is not known or the CPU data is not available. Do not stop the import.
        return
    missing = [f for f in _REQUIRED if f not in features]
    if missing:
        raise ImportError(
            "manyfaced needs an x86-64-v3 CPU (AVX2, FMA, BMI2: Intel Haswell 2013+ or AMD Zen "
            f"2017+). This CPU does not have {', '.join(missing)}. "
            "To disable this check, set MANYFACED_SKIP_CPU_CHECK=1."
        )

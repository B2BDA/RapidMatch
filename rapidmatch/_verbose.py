"""Opt-in stage lines for a matching run.

Independent of progress bars. Default off. When on, timestamped lines go to
stderr so they do not mix with printed results. rewrite() overwrites the
current line for live counters; emit() always ends with a newline.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Optional, TextIO


def rss_mb() -> Optional[float]:
    """Current process RSS in MiB, or None if the OS will not say."""
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss > 1e7:
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        pass
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


class VerboseLog:
    """Tiny stderr logger. No-op when enabled is False."""

    def __init__(
        self,
        enabled: bool = False,
        stream: Optional[TextIO] = None,
        clock: Any = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock if clock is not None else time.monotonic
        self._t0 = self._clock()
        self._open = False

    def emit(self, stage: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self._write(stage, fields, newline=True)

    def rewrite(self, stage: str, **fields: Any) -> None:
        if not self.enabled:
            return
        self._write(stage, fields, newline=False)

    def _write(self, stage: str, fields: dict[str, Any], newline: bool) -> None:
        elapsed = self._clock() - self._t0
        parts = ["rapidmatch", "t=%.1fs" % elapsed, stage]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append("%s=%s" % (key, value))
        line = " ".join(parts)
        prefix = chr(13) if self._open else ""
        suffix = chr(10) if newline else ""
        try:
            self.stream.write(prefix + line + suffix)
            self.stream.flush()
        except Exception:
            self._open = False
            return
        self._open = not newline

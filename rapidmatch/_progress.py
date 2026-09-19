"""Opt-in progress bars for terminals and Jupyter.

Bars appear only when the user opts in (`config.progress=True`) AND tqdm is
installed (the `progress` extra) AND there is somewhere to draw them:

- **Terminal** (stderr is a TTY): classic `tqdm.std` bars on stderr.
- **Jupyter / IPython**: live `tqdm.notebook` widgets via `display(update=...)`.
- **Anything else** (pipelines, captured stderr, daemons): every helper is a
  no-op so call sites stay branch-free.

When `progress=True` but a bar cannot be drawn, a one-line hint is printed to
stderr (once per process) naming the exact reason instead of failing silently.
You can self-check rendering with:

    uv run python -c "from rapidmatch._progress import _probe; _probe()"
"""

from __future__ import annotations

import sys
from typing import Any, Iterable, Optional, TypeVar

T = TypeVar("T")

_stderr_tty: Optional[bool] = None
_warned: bool = False


def _is_tty() -> bool:
    global _stderr_tty
    if _stderr_tty is None:
        try:
            _stderr_tty = bool(sys.stderr.isatty())
        except Exception:
            _stderr_tty = False
    return _stderr_tty


def _environment() -> str:
    """'terminal' (live TTY), 'notebook' (IPython/Jupyter), else 'none'."""
    if _is_tty():
        return "terminal"
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except (NameError, AttributeError):
        shell = ""
    if shell in ("ZMQInteractiveShell", "TerminalInteractiveShell"):
        return "notebook"
    return "none"


def _emit_hint(message: str) -> None:
    """Print a diagnostic once per process; never raise."""
    global _warned
    if _warned:
        return
    _warned = True
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass


def _backend() -> Any:
    """The tqdm class that fits the current environment best."""
    if _environment() == "notebook":
        try:
            from tqdm.notebook import tqdm as nb_tqdm  # type: ignore[import-not-found]

            return nb_tqdm
        except Exception:
            pass
    from tqdm.std import tqdm  # optional extra, imported lazily

    return tqdm


def _build_bar(*args: Any, **kwargs: Any) -> Any:
    """Construct a live bar, cascading notebook -> terminal -> no-op.

    In notebooks we prefer a real `tqdm.notebook` widget; if it cannot render
    (e.g. no `ipywidgets`, headless kernel), fall back to a plain terminal
    style bar on stderr rather than silently disappearing.
    """
    if _environment() == "notebook":
        try:
            import ipywidgets  # noqa: F401  no widget machinery -> skip cleanly
        except Exception:
            pass
        else:
            try:
                from tqdm.notebook import tqdm as nb_tqdm  # type: ignore[import-not-found]

                return nb_tqdm(*args, **kwargs)
            except Exception:
                pass
    try:
        from tqdm.std import tqdm

        return tqdm(*args, **kwargs)
    except Exception:
        return _NullBar(*args, **kwargs)


def _bar_kwargs(desc: str, total: Optional[int]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"desc": desc, "total": total, "miniters": 1}
    if _environment() == "terminal":
        kwargs["file"] = sys.stderr
        kwargs["dynamic_ncols"] = True
        kwargs["leave"] = True  # keep finished bars on screen
    return kwargs


def enabled(progress: bool) -> bool:
    """True when a real bar is possible; otherwise helpers no-op.

    A one-line hint is emitted to stderr (once) whenever bars were requested
    but cannot be drawn, so the reason is never silent.
    """
    if not progress:
        return False
    if _environment() == "none":
        _emit_hint(
            "rapidmatch: progress bars disabled - stderr is not a TTY and not "
            "running inside Jupyter/IPython, so there is nowhere to draw. Run "
            "from a real terminal or from a notebook to see bars."
        )
        return False
    try:
        _backend()
    except Exception:
        _emit_hint(
            "rapidmatch: progress bars disabled - tqdm is not installed. "
            'Install the extra with: uv add "rapidmatch[progress]" '
            '(or: pip install "rapidmatch[progress]").'
        )
        return False
    return True


class _NullBar:
    """Swallow all calls so `enabled=False` code paths stay identical."""

    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        total: Optional[int] = None,
        desc: str = "",
        **_kwargs: Any,
    ) -> None:
        self._iterable = iterable

    def update(self, n: int = 1) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass

    def set_description(self, *_: Any, **__: Any) -> None:
        pass

    def set_postfix(self, *_: Any, **__: Any) -> None:
        pass

    def refresh(self) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def __iter__(self) -> Iterable[T]:
        return iter(self._iterable) if self._iterable is not None else iter(())


def iterable_bar(
    iterable: Iterable[T],
    desc: str = "",
    enabled_flag: bool = False,
    total: Optional[int] = None,
) -> Any:
    """Wrap `iterable` with a bar, or return a no-op wrapper."""
    if not enabled(enabled_flag):
        return _NullBar(iterable=iterable)
    return _build_bar(iterable, **_bar_kwargs(desc, total))


def manual_bar(
    total: Optional[int] = None,
    desc: str = "",
    enabled_flag: bool = False,
) -> Any:
    """A bar advanced by explicit `update(...)` calls."""
    if not enabled(enabled_flag):
        return _NullBar(total=total, desc=desc)
    return _build_bar(**_bar_kwargs(desc, total))


def _track(bar: Any) -> Any:
    """Return a callable that advances `bar`, keeping its name stable.

    The current stage is shown as a postfix (`stage=...`) so the bar's title
    (e.g. "pipeline") never gets relabeled out from under the user.
    """

    def advance(label: str) -> None:
        bar.set_postfix(stage=label)
        bar.update(1)

    return advance


def _probe() -> None:
    """Draw a disposable progress bar so rendering can be verified in place.

    Run with:

        uv run python -c "from rapidmatch._progress import _probe; _probe()"

    (A `python -m rapidmatch._progress` entry also works, but warns because the
    package init already imported this module.) If bars cannot be drawn here, a
    hint on stderr explains why.
    """
    import time

    bar = manual_bar(total=100, desc="probe", enabled_flag=True)
    try:
        for _ in range(100):
            bar.update(1)
            time.sleep(0.03)
        bar.set_description("probe done")
    finally:
        bar.close()


if __name__ == "__main__":
    _probe()
"""Progress-bar gating: opt-in, environment-aware, never breaks the pipeline."""

import builtins

import pytest

from rapidmatch._progress import (
    _NullBar,
    _bar_kwargs,
    _environment,
    enabled,
    iterable_bar,
    manual_bar,
)


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    monkeypatch.setattr("rapidmatch._progress._stderr_tty", None)
    monkeypatch.setattr("rapidmatch._progress._warned", False)
    yield


def _set_tty(monkeypatch, value):
    monkeypatch.setattr("rapidmatch._progress._is_tty", lambda: value)


def _set_ipython(monkeypatch, present=True):
    if present:
        shell = type("ZMQInteractiveShell", (object,), {})()
        monkeypatch.setitem(builtins.__dict__, "get_ipython", lambda: shell)
    else:
        monkeypatch.setitem(builtins.__dict__, "get_ipython", lambda: None)


def test_progress_opt_out_is_never_enabled(monkeypatch):
    _set_tty(monkeypatch, True)
    assert not enabled(False)
    assert isinstance(manual_bar(total=5, enabled_flag=False), _NullBar)
    assert isinstance(iterable_bar(range(3), enabled_flag=False), _NullBar)


def test_progress_disabled_outside_terminal_and_notebook(monkeypatch):
    _set_tty(monkeypatch, False)
    _set_ipython(monkeypatch, present=False)
    assert _environment() == "none"
    assert not enabled(True)
    assert isinstance(manual_bar(total=5, enabled_flag=True), _NullBar)


def test_progress_enabled_in_terminal(monkeypatch):
    _set_tty(monkeypatch, True)
    _set_ipython(monkeypatch, present=False)
    assert _environment() == "terminal"
    assert enabled(True)
    bar = manual_bar(total=3, desc="t", enabled_flag=True)
    assert not isinstance(bar, _NullBar)
    bar.update(1)
    bar.close()
    assert _bar_kwargs("d", 3)["file"] is not None  # terminal bars go to stderr


def test_progress_enabled_in_notebook(monkeypatch):
    _set_tty(monkeypatch, False)
    _set_ipython(monkeypatch, present=True)
    assert _environment() == "notebook"
    assert enabled(True)
    # Headless kernels degrade to a terminal-style bar; never a silent no-op.
    bar = manual_bar(total=3, desc="nb", enabled_flag=True)
    assert not isinstance(bar, _NullBar)
    bar.update(1)
    bar.close()


def _block_tqdm(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] == "tqdm":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_progress_never_breaks_when_tqdm_missing(monkeypatch):
    _set_tty(monkeypatch, True)
    _block_tqdm(monkeypatch)
    assert not enabled(True)
    assert isinstance(manual_bar(total=5, enabled_flag=True), _NullBar)


def test_hint_when_tqdm_missing(monkeypatch, capsys):
    _set_tty(monkeypatch, True)
    _block_tqdm(monkeypatch)
    assert not enabled(True)
    assert "tqdm is not installed" in capsys.readouterr().err
    enabled(True)  # hint fires once per process
    assert capsys.readouterr().err == ""


def test_hint_when_no_display(monkeypatch, capsys):
    _set_tty(monkeypatch, False)
    _set_ipython(monkeypatch, present=False)
    assert not enabled(True)
    assert "not a TTY" in capsys.readouterr().err


def test_no_hint_when_flag_off(monkeypatch, capsys):
    _set_tty(monkeypatch, False)
    _set_ipython(monkeypatch, present=False)
    assert not enabled(False)
    assert capsys.readouterr().err == ""


def test_no_hint_when_renderable(monkeypatch, capsys):
    _set_tty(monkeypatch, True)
    _set_ipython(monkeypatch, present=False)
    assert enabled(True)
    assert capsys.readouterr().err == ""
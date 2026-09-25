import io

from rapidmatch._verbose import VerboseLog, rss_mb


def test_verbose_off_writes_nothing() -> None:
    buf = io.StringIO()
    log = VerboseLog(enabled=False, stream=buf)
    log.emit("start")
    log.rewrite("match", assigned=3)
    log.emit("done")
    assert buf.getvalue() == ""


def test_verbose_on_writes_stage_line() -> None:
    buf = io.StringIO()
    log = VerboseLog(enabled=True, stream=buf)
    log.emit("stratify", strata=12, eligible=10, no_control=1, thin=2)
    text = buf.getvalue()
    assert "rapidmatch" in text
    assert "stratify" in text
    assert "strata=12" in text
    assert "eligible=10" in text
    assert text.endswith("\n")


def test_verbose_rewrite_then_emit_ends_with_newline() -> None:
    buf = io.StringIO()
    log = VerboseLog(enabled=True, stream=buf)
    log.rewrite("match", assigned=4)
    log.rewrite("match", assigned=9)
    log.emit("match", assigned=9)
    text = buf.getvalue()
    assert "assigned=4" in text
    assert "assigned=9" in text
    assert text.endswith("\n")


def test_rss_mb_is_non_negative() -> None:
    value = rss_mb()
    assert value is None or value >= 0.0

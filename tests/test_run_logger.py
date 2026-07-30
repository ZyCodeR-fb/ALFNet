"""The console log must survive the terminal session that produced it.

`training_summary.json` keeps `validation_group_volatility` but drops the
per-epoch curve it was computed from, so the only record of how each lifecycle
group moved is stdout. Losing that once already cost a round of diagnosis.
"""

import sys
from types import SimpleNamespace

from utils.run_logger import install_run_log


def _restore(saved):
    sys.stdout, sys.stderr = saved


def test_install_run_log_writes_under_the_run_log_dir(tmp_path):
    log_dir = tmp_path / "outputs" / "ALFNet" / "run-x" / "logs"
    args = SimpleNamespace(log_dir=str(log_dir), test_for_async_lifecycle=False)
    saved = (sys.stdout, sys.stderr)
    try:
        path = install_run_log(args)
        assert path is not None
        print("epoch 1 val_lifecycle: unseen_mae=4.0")
        print("a warning", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        _restore(saved)

    assert log_dir.is_dir()
    written = (log_dir / __import__("os").path.basename(path)).read_text(encoding="utf-8")
    assert "unseen_mae=4.0" in written
    assert "a warning" in written, "stderr must land in the log too, for tracebacks"
    # named for the mode so a later test_async run cannot truncate it
    assert __import__("os").path.basename(path).startswith("train_")


def test_test_async_mode_gets_its_own_log_file(tmp_path):
    log_dir = tmp_path / "logs"
    args = SimpleNamespace(log_dir=str(log_dir), test_for_async_lifecycle=True)
    saved = (sys.stdout, sys.stderr)
    try:
        path = install_run_log(args)
    finally:
        _restore(saved)
    assert __import__("os").path.basename(path).startswith("test_async_")


def test_no_output_root_means_no_log_and_no_crash():
    """The legacy flat layout has no per-run home; it must not blow up."""
    saved = (sys.stdout, sys.stderr)
    try:
        assert install_run_log(SimpleNamespace()) is None
        assert install_run_log(SimpleNamespace(log_dir=None)) is None
    finally:
        _restore(saved)
    assert sys.stdout is saved[0], "stdout was replaced despite there being no log"


def test_tee_still_reaches_the_console(tmp_path, capsys):
    """It tees, not redirects: a tmux session must still show progress."""
    args = SimpleNamespace(log_dir=str(tmp_path / "logs"), test_for_async_lifecycle=False)
    saved = (sys.stdout, sys.stderr)
    try:
        path = install_run_log(args)
        print("visible")
        sys.stdout.flush()
    finally:
        _restore(saved)
    # capsys sees it because the tee wrote through to the captured stream
    assert "visible" in capsys.readouterr().out
    assert "visible" in open(path, encoding="utf-8").read()

"""Duplicate a run's console output into its own artefact directory.

`args.apply_output_root` has always derived `log_dir` as
`outputs/<model>/<run_id>/logs`, but nothing ever wrote there: the per-epoch
validation numbers only existed in the terminal. That made the one diagnostic
that matters -- how each lifecycle group moves from epoch to epoch -- impossible
to recover after the fact, because `training_summary.json` keeps the aggregated
`validation_group_volatility` and drops the underlying curve.

This tees rather than redirects, so a run watched in tmux still prints normally
while the same bytes land in a file that survives the session.
"""

import atexit
import os
import sys
from datetime import datetime


class _Tee:
    """Write to the console and the log file at once.

    Flushes on every write: a training run is long, and a log you cannot
    `tail -f` while it is running is much less useful than one you can.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()
        self._handle.write(text)
        self._handle.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    # Libraries probe these before deciding how to render progress output.
    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


def install_run_log(args, mode=None):
    """Tee stdout and stderr into `<log_dir>/<mode>_<timestamp>.log`.

    Returns the log path, or None when the run is not output-root scoped (the
    legacy flat layout has no per-run home to write into).
    """
    log_dir = getattr(args, "log_dir", None)
    if not log_dir:
        return None

    if mode is None:
        mode = "test_async" if getattr(args, "test_for_async_lifecycle", False) else "train"
    os.makedirs(log_dir, exist_ok=True)
    # Timestamped rather than fixed: re-running test_async over a finished
    # training run must not truncate the training log that explains it.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(log_dir, f"{mode}_{stamp}.log")

    # errors="replace" because a Windows console encoding is not utf-8 and a
    # stray glyph in a third-party warning should not kill a training run.
    handle = open(path, "w", encoding="utf-8", errors="replace")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, handle)
    sys.stderr = _Tee(original_stderr, handle)

    def _restore():
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if not handle.closed:
            handle.flush()
            handle.close()

    atexit.register(_restore)
    return path

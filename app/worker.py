from __future__ import annotations

import json
from PySide6.QtCore import QObject, Signal, Slot, QProcess


class ProcessWorker(QObject):
    """
    Run a python CLI in a separate process (subprocess-safe).
    - Streams stdout/stderr to log signal
    - Parses a single-line JSON meta emitted as: __META__{...}
    """

    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, args: list[str], cwd: str | None = None):
        super().__init__()
        self._args = args
        self._cwd = cwd
        self._proc: QProcess | None = None
        self._last_meta: dict | None = None
        self._stderr_buf: list[str] = []

    @Slot()
    def start(self):
        if self._proc is not None:
            self.failed.emit("Process already started.")
            return

        p = QProcess(self)
        self._proc = p

        if self._cwd:
            p.setWorkingDirectory(self._cwd)

        p.readyReadStandardOutput.connect(self._on_stdout)
        p.readyReadStandardError.connect(self._on_stderr)
        p.finished.connect(self._on_finished)

        program = self._args[0]
        arguments = self._args[1:]

        self.log.emit(" ".join(self._args))
        p.start(program, arguments)

        # start fail quickly
        if not p.waitForStarted(3000):
            self.failed.emit("Failed to start process.")
            return

    def kill(self):
        if self._proc is not None:
            self._proc.kill()

    def _on_stdout(self):
        assert self._proc is not None
        txt = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")

        for line in txt.splitlines():
            if line.startswith("__META__"):
                payload = line[len("__META__") :]
                try:
                    self._last_meta = json.loads(payload)
                except Exception:
                    self._last_meta = {"error": "Failed to parse meta JSON."}
            else:
                self.log.emit(line)

    def _on_stderr(self):
        assert self._proc is not None
        txt = bytes(self._proc.readAllStandardError()).decode("utf-8", errors="replace")
        for line in txt.splitlines():
            self._stderr_buf.append(line)
            self.log.emit("[stderr] " + line)

    def _on_finished(self, exit_code: int, _exit_status):
        meta = self._last_meta if self._last_meta is not None else {}

        if exit_code == 0 and (not meta.get("error")):
            self.done.emit(meta)
            return

        msg = meta.get("error") or f"Process failed. exit_code={exit_code}"
        if self._stderr_buf:
            msg += "\n\n[stderr]\n" + "\n".join(self._stderr_buf[-200:])
        self.failed.emit(msg)

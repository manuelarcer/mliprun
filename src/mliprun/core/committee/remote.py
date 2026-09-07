"""One committee member, running in its own MLIP env's interpreter.

Owns the subprocess: spawn, load handshake, request/response with a deadline,
and teardown. POSIX only -- ``select`` on a pipe does not work on Windows,
and the committee bridge is not supported there.

Design: docs/superpowers/specs/2026-09-04-committee-uncertainty-design.md
"""
from __future__ import annotations

import atexit
import logging
import os
import select
import signal
import subprocess
import textwrap
import time
import weakref

from mliprun.core.committee import protocol

logger = logging.getLogger(__name__)

#: Per-request budget. A UMA single point costs ~0.2 s and a large system
#: rather more; 300 s is "something is wrong", not "this is slow".
DEFAULT_CALC_TIMEOUT_S = 300.0

#: Model load can include a first-use checkpoint download (SevenNet ships
#: ~103 MB per checkpoint into site-packages; UMA pulls from Hugging Face),
#: so the load budget is far larger than the per-call one. On an air-gapped
#: node, pre-fetch the checkpoints per the install recipe rather than
#: raising this.
DEFAULT_LOAD_TIMEOUT_S = 1800.0

_QUIT_GRACE_S = 5.0

#: Live members, for the atexit backstop. Weak so a garbage-collected member
#: is not kept alive by the registry itself.
_LIVE: "weakref.WeakSet" = weakref.WeakSet()


class MemberError(RuntimeError):
    """A committee member failed.

    Carries the member's name, the path to its worker log, and the remote
    traceback, because the driver has no other window into that env.
    """

    def __init__(self, member: str, message, log_path=None,
                 remote_traceback: str = ""):
        self.member = member
        self.log_path = str(log_path) if log_path else None
        self.remote_traceback = remote_traceback or ""
        text = f"committee member '{member}' failed: {message}"
        if self.log_path:
            text += f"\n  worker log: {self.log_path}"
        if self.remote_traceback:
            text += ("\n  remote traceback:\n"
                     + textwrap.indent(self.remote_traceback.rstrip(), "    "))
        super().__init__(text)


@atexit.register
def _kill_leftover_members() -> None:
    """Last-resort teardown for anything still alive at interpreter exit.

    cos-cluster has no scheduler to reap orphans, and a leaked worker holds a
    CUDA context that makes the GPU look busy to everyone else on the node.
    """
    for member in list(_LIVE):
        try:
            member.close(timeout=0.5)
        except Exception:  # noqa: BLE001 -- nothing useful to do at exit
            pass


class RemoteMember:
    """One MLIP env's worker process, addressed as a calculator.

    Parameters
    ----------
    name : str
        Stable identifier. Becomes a CSV column header and the label in
        every error message, so keep it short and filesystem-safe.
    python_exe : str or Path
        The member env's interpreter -- ``<env>/bin/python``.
    mlip : str
        MLIP tag for that env, or ``"emt"`` for the reserved test member.
    uma_task, mace_head, sevennet_task : str, optional
        Head/task selectors, forwarded verbatim. Independent fine-tunes with
        independent energy zeros (CANON C1/C3): never defaulted here.
    device : str
        ``"auto"``, ``"cuda"`` or ``"cpu"``, resolved inside the member env
        (the driver may have no torch at all).
    gpu : int, optional
        Sets ``CUDA_VISIBLE_DEVICES`` for this member's process only. Omitted,
        the member inherits the driver's environment.
    log_path : str or Path, optional
        File that receives the worker's stderr -- library banners, remote
        tracebacks. ``None`` inherits the driver's stderr.
    timeout : float
        Per-request deadline in seconds.
    load_timeout : float
        Deadline for the one-time model load.
    argv : list of str, optional
        Full command line for the worker. Defaults to
        ``[python_exe, "-m", "mliprun.core.committee.worker"]``. Override it
        to wrap the interpreter in a launcher (``srun``, a module-load shim),
        or to point at a stub in tests.
    """

    def __init__(self, name, python_exe, *, mlip, uma_task=None,
                 mace_head=None, sevennet_task=None, device="auto", gpu=None,
                 log_path=None, timeout=DEFAULT_CALC_TIMEOUT_S,
                 load_timeout=DEFAULT_LOAD_TIMEOUT_S, argv=None):
        self.name = str(name)
        self.python_exe = str(python_exe)
        self.mlip = mlip
        self.uma_task = uma_task
        self.mace_head = mace_head
        self.sevennet_task = sevennet_task
        self.device = device
        self.gpu = gpu
        self.log_path = str(log_path) if log_path else None
        self.timeout = float(timeout)
        self.load_timeout = float(load_timeout)
        self.argv = (list(argv) if argv else
                     [self.python_exe, "-m", "mliprun.core.committee.worker"])
        #: Kept after teardown so a caller can still say which process ran.
        self.pid = None
        self.versions: dict = {}
        self._proc = None
        self._log = None
        self._buffer = bytearray()

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> dict:
        """Spawn the worker and load its model. Returns the versions block.

        Raises
        ------
        MemberError
            If the interpreter is missing, mliprun is not installed in that
            env, the MLIP package is absent, or the model refuses to load.
            Every one of these surfaces here -- before the optimizer starts.
        """
        env = dict(os.environ)
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        # Unbuffered, so a worker that dies mid-import still leaves its
        # traceback in the log.
        env["PYTHONUNBUFFERED"] = "1"

        if self.log_path:
            try:
                self._log = open(self.log_path, "wb")
            except OSError as exc:
                raise MemberError(self.name,
                                  f"could not open worker log: {exc}") from exc

        try:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log if self._log is not None else None,
                env=env,
                close_fds=True,
                # Own process group, so teardown can kill the worker and
                # anything it spawned (torch dataloader workers) in one go.
                start_new_session=True,
            )
        except OSError as exc:
            # The log was already opened above; release it deterministically
            # rather than leaving it for GC -- Task 5 closes only the
            # members that started successfully, never this one.
            self._close_log()
            raise MemberError(
                self.name,
                f"could not start worker: {exc} (command: {' '.join(self.argv)})",
                self.log_path) from exc

        self.pid = self._proc.pid
        _LIVE.add(self)
        response = self._exchange(
            protocol.load_request(self.mlip, uma_task=self.uma_task,
                                  mace_head=self.mace_head,
                                  sevennet_task=self.sevennet_task,
                                  device=self.device),
            self.load_timeout)
        self.versions = response.get("versions", {})
        return self.versions

    def close(self, timeout: float = _QUIT_GRACE_S) -> None:
        """Shut the worker down. Idempotent, and safe on a half-started member."""
        proc = self._proc
        if proc is None:
            self._close_log()
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(protocol.encode(protocol.quit_request()))
                    proc.stdin.flush()
                except Exception:  # noqa: BLE001 -- already gone; escalate below
                    pass
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._kill()
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
            self._close_log()
            self._proc = None
            _LIVE.discard(self)

    def _close_log(self) -> None:
        if self._log is not None:
            try:
                self._log.close()
            except Exception:  # noqa: BLE001
                pass
            self._log = None

    def _kill(self) -> None:
        """SIGKILL the worker's whole process group, then reap it."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("committee member '%s' (pid %s) would not die",
                           self.name, proc.pid)

    # -- requests ----------------------------------------------------------

    def calculate(self, numbers, positions, cell, pbc):
        """Single-point this geometry. Returns ``(energy, forces)``.

        ``forces`` comes back as a nested list; the caller converts. Raw
        forces, with no constraints applied -- those stay driver-side.
        """
        response = self._exchange(
            protocol.calc_request(numbers, positions, cell, pbc), self.timeout)
        try:
            energy = float(response["energy"])
            forces = response["forces"]
        except (KeyError, TypeError, ValueError) as exc:
            raise MemberError(self.name, f"malformed calc response: {exc}",
                              self.log_path) from exc
        return energy, forces

    def _exchange(self, request: dict, timeout: float) -> dict:
        if not self.is_alive:
            raise MemberError(self.name, "worker is not running "
                              "(start() not called, or already closed)",
                              self.log_path)
        try:
            self._proc.stdin.write(protocol.encode(request))
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise MemberError(self.name, f"could not send request: {exc}",
                              self.log_path) from exc

        line = self._read_line(timeout)
        try:
            response = protocol.decode(line)
        except protocol.ProtocolError as exc:
            raise MemberError(self.name, str(exc), self.log_path) from exc
        if not response.get("ok"):
            raise MemberError(self.name,
                              response.get("error", "unknown error"),
                              self.log_path,
                              response.get("traceback", ""))
        return response

    def _read_line(self, timeout: float) -> bytes:
        """Read one newline-terminated line, or fail at the deadline.

        ``select`` rather than a blocking ``readline``: a hung member must
        fail, not stall the run overnight. Leftover bytes past the newline
        stay buffered for the next call, so a worker that writes two lines in
        one chunk is handled correctly.
        """
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise MemberError(
                    self.name,
                    f"no response within {timeout:g} s (worker killed)",
                    self.log_path)
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                # EOF on the pipe means the worker's fds were torn down, which
                # can be observed here microseconds before waitpid() reflects
                # its exit status (confirmed racy on macOS: a bare poll()
                # reads None here in ~90% of trials). A short blocking wait
                # closes that window instead of reporting a useless
                # "returncode=None".
                try:
                    code = self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    code = self._proc.poll()
                raise MemberError(
                    self.name,
                    f"worker exited (returncode={code}) before replying",
                    self.log_path)
            self._buffer.extend(chunk)

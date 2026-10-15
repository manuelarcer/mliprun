"""Scripted stand-in for ``python -m mliprun.webapp.worker <run_dir>``.

Backend tests launch this instead of the real worker so queue, cancel and
crash behaviour can be tested quickly without ASE calculations or MLIPs.
It honours the worker process contract (PLAN 8.1): it appends events to
``events.jsonl`` with ``seq`` = line index, writes ``summary.json``, exits
0 / 1 / 3, handles SIGTERM, and never writes ``job_status``.

Behaviour is scripted through the job label (``"steps=5;sleep=0.1"``), with
``FAKE_WORKER`` in the environment as a fallback using the same syntax:

- ``steps=N``      number of ``opt_step`` events (default 3)
- ``sleep=S``      seconds between steps (default 0.01)
- ``exit=N``       exit with code N after the steps, no result
- ``error=MSG``    emit an ``error`` event, write a failed summary, exit 1
- ``crash=1``      ``os._exit(7)`` right after the first event (no summary)
- ``ignore_term=1`` ignore SIGTERM (to test the SIGKILL fallback)
- ``no_summary=1`` exit 0 after the result without writing summary.json
- ``partial=1``    leave a partial (unterminated) line at the end of events.jsonl

It also appends ``{"start": t0, "end": t1, "pid": pid}`` lines to
``<run_dir>/fake_times.jsonl`` so tests can check that jobs never overlap.
Standard library only, so it starts fast.
"""
import json
import os
import signal
import sys
import time


class Cancelled(Exception):
    pass


def parse_script(text):
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def main():
    run_dir = sys.argv[-1]
    t0 = time.time()
    with open(os.path.join(run_dir, "job.json")) as fh:
        job = json.load(fh)
    script = parse_script(os.environ.get("FAKE_WORKER", ""))
    script.update(parse_script(job.get("label") or ""))
    steps = int(script.get("steps", 3))
    pause = float(script.get("sleep", 0.01))

    if script.get("ignore_term") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        def on_term(signum, frame):
            raise Cancelled()
        signal.signal(signal.SIGTERM, on_term)

    path = os.path.join(run_dir, "events.jsonl")
    with open(path, "rb") as fh:
        seq = fh.read().count(b"\n")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    state = {"seq": seq, "first": True}

    def emit(etype, **payload):
        event = {"job_id": job["job_id"], "seq": state["seq"], "t": time.time(),
                 "type": etype}
        event.update(payload)
        os.write(fd, (json.dumps(event) + "\n").encode())
        state["seq"] += 1
        if state["first"] and script.get("crash") == "1":
            os._exit(7)
        state["first"] = False

    def summary(status, result=None, error=None):
        data = {"job_id": job["job_id"], "calc_type": job["calc_type"], "status": status,
                "started_at": None, "finished_at": None, "walltime_s": time.time() - t0,
                "result": result, "error": error,
                "reproduce": {"status": "no", "notes": ["fake worker"]}}
        with open(os.path.join(run_dir, "summary.json"), "w") as fh:
            json.dump(data, fh)

    def record_times():
        with open(os.path.join(run_dir, "fake_times.jsonl"), "a") as fh:
            fh.write(json.dumps({"start": t0, "end": time.time(), "pid": os.getpid()}) + "\n")

    code = 0
    try:
        emit("phase", phase="setup", message="fake worker starting")
        with open(os.path.join(run_dir, "reproduce.json"), "w") as fh:
            json.dump({"status": "no", "notes": ["fake worker"], "argv": []}, fh)
        emit("phase", phase="run", message="running")
        for i in range(steps):
            frame = {"positions": [[0.0, 0.0, float(i)]], "cell": [[0.0] * 3] * 3}
            emit("opt_step", step=i, energy=-1.0 - 0.1 * i, fmax=1.0 / (i + 1), frame=frame)
            time.sleep(pause)
        if "error" in script:
            emit("error", message=script["error"], exception_type="RuntimeError",
                 traceback="Traceback: fake")
            summary("failed", error={"message": script["error"],
                                     "exception_type": "RuntimeError"})
            code = 1
        elif "exit" in script:
            code = int(script["exit"])
        else:
            emit("result", calc_type=job["calc_type"], record_status="converged",
                 record_results={}, walltime_s=time.time() - t0, converged=True,
                 final_energy_eV=-1.0 - 0.1 * max(steps - 1, 0), final_fmax_eV_per_A=0.01,
                 steps=steps, final_frame={"positions": [[0.0, 0.0, 0.0]],
                                           "cell": [[0.0] * 3] * 3})
            if script.get("no_summary") != "1":
                summary("done", result={"converged": True})
        if script.get("partial") == "1":
            os.write(fd, b'{"job_id": "partial')
    except Cancelled:
        emit("log", level="info", message="cancelled by user", logger="fake")
        summary("cancelled")
        code = 3
    finally:
        record_times()
        os.close(fd)
    sys.exit(code)


if __name__ == "__main__":
    main()

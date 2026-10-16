# Remote machines for the web app (contract)

Goal: the web app (server stays on the laptop) can run a job on another machine
over SSH. The user defines machines and Python environments in the UI; nothing
is preconfigured. Each submission picks a *target*: this computer (`local`) or
one remote environment. Juan's rulings (2026-10-16) are marked **[J]**.

## 1. Concepts and persistence

- **Machine**: SSH connection settings. **Environment (env)**: one Python
  interpreter on a machine (one MLIP venv). A target is `"local"` or an env id.
  **[J]** one selectable entry per venv; the default env name is
  `<machine name>-<venv dir>` (python `/x/.venv/mace/bin/python` -> venv dir
  `mace`; if the path has no `bin/` parent, use the python file's parent dir
  name), editable by the user.
- **[J]** Stored across restarts in a JSON file, default
  `~/.mliprun/webapp-machines.json`, overridable with `mlip webapp
  --machines-file PATH` and `create_app(machines_file=...)`. Written atomically
  (temp file + `os.replace`), mode 0600. Schema:

```json
{"schema_version": 1,
 "machines": [{"id": "m-<6 hex>", "name": "cos-cluster", "host": "cos-cluster",
   "user": null, "port": null, "identity_file": null, "jump_host": null,
   "default_location": "/scratchb/juar/webapp-runs",
   "poll_interval_s": 2.0, "copy_back_max_mb": 50,
   "envs": [{"id": "e-<6 hex>", "name": "cos-cluster-mace",
     "python": "/scratchb/.../.venv/mace/bin/python", "setup": "",
     "probe": {...last probe result or null...}}]}]}
```

- Validation (these values reach an `ssh` argv; reject, never escape):
  `name` 1-60 chars, `[A-Za-z0-9._+-]`, unique among machines / among all envs;
  `host` `^[A-Za-z0-9][A-Za-z0-9._-]*$`; `user` `^[A-Za-z0-9._-]+$` or null;
  `port` 1-65535 or null; `identity_file` absolute or `~` path, no newline/NUL;
  `jump_host` `^([A-Za-z0-9._-]+@)?[A-Za-z0-9][A-Za-z0-9._-]*(:[0-9]+)?$`;
  `default_location` absolute remote path (may start with `~`), required;
  `poll_interval_s` 0.5-60 (default 2.0); `copy_back_max_mb` 0-10000 (default
  50); env `python` absolute path, no whitespace/newline/NUL; env `setup` free
  shell text up to 4000 chars (runs before every remote command for that env:
  `module load`, `export CUDA_VISIBLE_DEVICES=1`; the app has no auth and binds
  to loopback, so this is the user's own shell, documented as such).
- A machine or env with non-terminal jobs cannot be deleted (409).

## 2. SSH transport (`webapp/remote.py`)

- Only the system `ssh` binary (no new Python dependency, honours
  `~/.ssh/config`). Argv: `[ssh_bin, "-o", "BatchMode=yes", "-o",
  "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o",
  "ServerAliveCountMax=3", "-o", "ControlMaster=auto", "-o",
  "ControlPath=<ctl_dir>/%C", "-o", "ControlPersist=300", (-p port) (-i file)
  (-J jump) (-l user), "--", host, remote_command]`. `ctl_dir` =
  `/tmp/mliprun-ssh-<uid>` (mode 0700; short because a unix socket path is
  limited to ~104 chars on macOS). `ssh_bin` defaults to `"ssh"`; tests pass a
  fake via `create_app(ssh_command=[...])`.
- `BatchMode=yes` means a host that needs a password fails fast; the error text
  says: "SSH needs key-based login (or an open ControlMaster); passwords cannot
  be typed by the web app".
- Every remote command is
  `/bin/sh -c <quoted script>` where script = `<env setup>\nexec <python> -m
  mliprun.webapp.remote_agent <subcommand> <quoted args>`; every value goes
  through `shlex.quote`. Exit code 255 = SSH-level failure (unreachable),
  reported as `RemoteUnreachable`; other non-zero = agent error, its stderr
  last line is the message.
- Calls are blocking (`subprocess.run` with timeout); the async server calls
  them through `asyncio.to_thread` / `run_in_threadpool`, never on the loop.

## 3. Remote agent (`webapp/remote_agent.py`)

Runs on the remote with the env's Python. **Standard library only** (it is
started on every poll, so import time matters; `mliprun/__init__.py` and
`mliprun/webapp/__init__.py` import nothing). Output: one JSON object on
stdout (except `poll`, `pack`, `cat`, see below). Subcommands:

| subcommand | does |
|---|---|
| `probe` | `{"agent_version", "mliprun_version", "python", "hostname", "worker_ok", "calculators": <list_calculators() after detect_devices()>}`. `worker_ok` = `import mliprun.webapp.worker` succeeds (done in a child so a heavy import never breaks the probe). Calculators come from `mliprun.webapp.calculators` (may import ase/typer; fine, probe is rare). |
| `gpus` | `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits` parsed to a list; `[]` with a note when nvidia-smi is missing. Informational only **[J: no automatic pre-flight]**. |
| `inspect LOCATION [FOLDER ...]` | same payload as `POST /api/paths/check` (`location, exists, writable, error, folders[]`), computed with the shared path helpers (section 4). |
| `create LOCATION FOLDER` | `mkdir -p LOCATION`, then **exclusive** `mkdir LOCATION/FOLDER` (exists -> exit 3, message "... already exists"), then extracts a tar stream from stdin into it (safe extraction: no absolute paths, no `..`, no links; `tarfile` `filter="data"` when available). Prints `{"run_dir": abs path}`. |
| `launch RUN_DIR` | reads the current `events.jsonl` content from stdin and writes it (replacing), then starts `remote_agent supervise RUN_DIR` detached (`start_new_session=True`, stdin DEVNULL, stdout/stderr -> `RUN_DIR/supervisor.log`, `close_fds=True`) so the ssh session returns at once; waits up to 15 s for `RUN_DIR/remote_pids.json`; prints it. |
| `supervise RUN_DIR` | (detached) starts `[sys.executable, -m, mliprun.webapp.worker, RUN_DIR]` with cwd RUN_DIR, stdout+stderr appended to `worker.log`, env `MPLBACKEND=Agg PYTHONUNBUFFERED=1`, `start_new_session=True`; writes `remote_pids.json` `{"supervisor_pid", "worker_pid", "hostname", "started_at"}`; ignores SIGTERM/SIGHUP itself; waits; writes `worker_exit.json` `{"returncode", "finished_at"}` atomically. |
| `poll RUN_DIR OFFSET MAX_BYTES` | first stdout line: JSON header `{"size", "alive", "exit": <worker_exit.json or null>, "chunk_len"}`; then exactly `chunk_len` raw bytes of `events.jsonl` from OFFSET, cut at the last newline within MAX_BYTES. `alive` = worker pid exists, not a zombie, and its `/proc/<pid>/cmdline` contains RUN_DIR (pid reuse guard). |
| `cancel RUN_DIR [--kill]` | SIGTERM (or SIGKILL) to the worker's process group after the same cmdline check. Prints `{"signalled": bool}`. |
| `list RUN_DIR` | `[{"path", "size", "mtime"}]`, same rules as the server's `_list_files` (symlinks pointing outside are skipped). |
| `pack RUN_DIR MAX_FILE_BYTES` | tar stream on stdout of every regular file <= MAX_FILE_BYTES except `job.json`, `state.json`, `events.jsonl`, `inputs/**` (the local copy is authoritative for those). |
| `cat RUN_DIR RELPATH` | raw bytes of one file inside RUN_DIR (path checked to stay inside). |

`AGENT_VERSION = 1`. The server refuses an env whose probe reports no
`remote_agent` (old mliprun) with: "mliprun on <machine> is too old for remote
runs: update it (git pull) in that environment".

## 4. Shared path helpers

Move `slugify_folder`, `folder_problem`, `suffix_width`, `suffixed_folder`,
`inspect_location`, `MAX_FOLDER_LEN`, `RESERVED_FOLDERS` from `jobs.py` into a
new stdlib-only module `webapp/paths.py`; `jobs.py` re-exports them (existing
imports keep working). The agent imports `paths.py`, so local and remote
checks are the same code. On the remote, `reserved` is empty.

## 5. Jobs with a remote target

- **[J] Two locations.** A remote job has a *remote run dir*
  `<remote location>/<folder>` where the worker runs, and a *local copy*
  `<local location>/<folder>` (local location default: the runs dir; same
  folder name). `JobRecord.run_dir` stays the local copy, so every existing
  reader (events, files, summary, zip) keeps working on local files.
- `job.json` gains (schema_version 2 for remote jobs; local jobs keep schema_version 1 and
  carry none of these keys, so old files and local jobs read as local):
  `"target": {"kind": "remote", "machine_id", "machine_name", "env_id",
  "env_name", "host", "python"}` (local jobs: `{"kind": "local"}`),
  `"remote_run_dir"`, `"location"` (remote), `"local_location"`.
- **Submission** (`POST /api/jobs`): new optional keys `target` (default
  `"local"`) and `local_location`. For a remote target: validate the
  calculator against the env's cached probe (not the server's own install;
  CUDA check against the probe's `cuda_available`), check the local copy
  folder does not exist, create the local copy exactly as today, then call
  `create` on the remote with a tar of `job.json` + `inputs/`. Folder exists
  remotely -> 422 `folder_exists` on the folder field; unreachable -> 422 on
  `target` with the SSH message. Any failure rolls back everything created by
  this submission (local dirs, and remote dirs via a `rmtree` of only the
  folders this request created: add agent subcommand `discard RUN_DIR` that
  deletes RUN_DIR only if it contains no `remote_pids.json`).
- `POST /api/paths/check`: new optional keys `target`, `local_location`. For a
  remote target the `location`/`folders` part comes from the agent `inspect`
  (folder `exists` = exists remotely **or** locally under the local location),
  plus `local_location: {location, exists, writable, error}`. Unreachable ->
  200 with `error` set to the SSH message (the form shows it; it is not a 5xx).
- `GET /api/calculators?target=<id>`: for an env, the cached probe's
  calculators payload (404 for an unknown id, 409 with a message when the env
  was never probed successfully).

## 6. Queues and the remote runner

- **[J] One serial queue per machine** (`local` is one queue). Jobs of
  different envs on the same machine share that machine's queue. Implement as
  queue key = `"local"` or machine id; `JobRegistry.pending` stays one FIFO
  list, `pop_next_pending(queue_key)` takes the first pending job of that
  key; `queue_position` counts within the key. One runner per key, created on
  demand (a machine added later gets a runner on its first job); `on_enqueue`
  wakes all runners. `running_job()` callers that assume one running job must
  be updated (`/api/health` gains `running_jobs: [...]`, keep `running_job`
  = the local one for compatibility).
- `RemoteQueueRunner` (same interface as `QueueRunner`), per job:
  1. `transition(running)` (server writes the seq-1 line locally).
  2. `launch` with the local `events.jsonl` on stdin (so seq continues on the
     remote: the worker's `EventWriter` counts existing lines). Set
     `job.offset`/local offset of the mirror to that byte length. Store
     `remote_pids` in `state.json` (`state["remote"] = {"pids": ..., "last_ok_at",
     "error"}`). Launch failure -> failed with the SSH/agent message.
  3. While the remote worker is alive the server must not write
     `events.jsonl` itself: generalise the guard in `_emit_status` from
     `job.proc is not None` to "a worker (local process or remote) is active".
  4. Every `poll_interval_s`: `poll` from the mirrored byte offset with
     `MAX_BYTES = 1 MiB`, append the chunk to the local `events.jsonl`, then
     `registry.drain(job)`. If `chunk_len == MAX_BYTES`-ish (more pending) poll
     again immediately. **[J] traffic**: no persistent stream, one short
     request per interval over the shared connection; only new bytes travel;
     structure frames are already throttled by "Structure every N".
  5. SSH failure: do **not** fail the job (the remote worker keeps running).
     Record `state.remote.error = "<message> (since <time>)"`, expose it in
     the job summary as `remote_error`, back off 2, 4, 8, ... 60 s, keep
     trying. Clear on the next success.
  6. Done when `exit` is non-null and the mirror reached `size` (or `alive`
     is false and no exit file: rc None, reason "remote worker disappeared").
     Then **[J] copy back small files**: `pack` with `copy_back_max_mb`
     -> extract into the local copy (same safe extraction), then `list` and
     write `remote_files.json` locally: `{"remote_host", "remote_run_dir",
     "files": [{"path", "size", "copied": bool}]}`. Copy-back failure is a
     log line, not a job failure. Then the usual `_finish(rc)` logic.
- Cancel: pending as today; running -> `cancel` (SIGTERM) in a thread, then
  `cancel --kill` after `cancel_grace_s`. If the remote is unreachable the
  request is remembered (`cancel_requested`) and delivered on reconnect.
- Server shutdown (`runner.stop()`): remote jobs are **not** cancelled
  (they keep running remotely); the runner task just stops.
- Server restart (`registry.load`): a `running` remote job is **reattached**
  (its runner resumes at step 4 from the local mirror's size), not failed.
  Pending remote jobs re-queue in their machine's queue. A job whose machine
  no longer exists in the machines file: stays as it is, `remote_error`
  "machine <name> no longer configured".

## 7. Files of a remote job

- `GET /api/jobs/{id}/files`: local files as today plus, from
  `remote_files.json`, entries with `"copied": false` returned as
  `{"path", "size", "mtime": null, "remote_only": true}`.
- `GET /api/jobs/{id}/files/{path}` for a remote-only path: stream it with
  agent `cat` (`StreamingResponse`, read the ssh stdout in chunks). Nothing is
  stored locally.
- The zip contains the local copy only; `remote_files.json` inside it says
  what stayed remote.
- JobSummary/JobDetail gain `target` (`{"kind", "machine_name", "env_name"}`),
  `remote_run_dir`, `remote_error`.

## 8. API for machines

`GET /api/machines` (envs nested, probes included) - `POST /api/machines` -
`PUT /api/machines/{mid}` - `DELETE /api/machines/{mid}` -
`POST /api/machines/{mid}/test` (runs `hostname` over SSH, then `gpus` through the first env; returns
`{ok, message, hostname, gpus, elapsed_s}`) - `POST /api/machines/{mid}/envs`
(`{python, name?, setup?}`; probes first; stores the env even if the probe
fails, with `probe: {"ok": false, "error": ...}`) - `PUT/DELETE
/api/machines/{mid}/envs/{eid}` - `POST /api/machines/{mid}/envs/{eid}/probe`.
All mutating routes keep the `X-Mliprun: 1` rule; 422 field errors use the
existing `validation_error` shape.

## 9. Front end

- Topbar button **Machines** opens a panel: machine list; add/edit/delete
  machine (fields of section 1); **Test connection** shows hostname, elapsed
  time, GPU table or the error; per machine an env list with **Add
  environment** (Python path; the name field fills with the default once the
  path is typed, editable), **Probe** (shows mliprun version, worker ok,
  calculators, CUDA), edit setup lines, delete.
- Form: **Run on** select first in the calculator block: "This computer" plus
  `<env name> (<machine name>)` for every env with a successful probe. Changing
  it reloads calculators/devices for that target. The last choice is
  remembered in localStorage (if it still exists).
- Remote target: the Location field is labelled **Remote location** (default:
  machine `default_location`, last value remembered per target) and a
  **Local copy location** field appears (default runs dir). The live path
  check and preview show the remote path and the local copy path.
- Job list/detail show the target name; detail shows remote run dir and a
  warning line with `remote_error` when set; remote-only files are marked
  "(on <machine>, not copied)" and still download.

## 10. Tests (no MLIP, no real SSH)

- `tests/webapp/fake_ssh.py`: executable stand-in for `ssh`. Skips options
  (`-o X`, `-p X`, `-i X`, `-J X`, `-l X`), then `--`, host, command; runs the
  command with `/bin/sh -c`, stdin/stdout passed through. Host `down` (or the
  file named by env `FAKE_SSH_DOWN` existing) -> exit 255 with an ssh-like
  message. Appends one line per call to the file in env `FAKE_SSH_LOG` (argv
  + bytes out) so tests can assert traffic. The env `python` in tests is
  `sys.executable`.
- Cover: machines file round trip and validation (each rejected field),
  default env name, ssh argv construction (`--` before host, quoting of a path
  with spaces and a `'`), agent subcommands called directly (create exclusive,
  unsafe tar rejected, poll cut at newline, cancel pid-reuse guard, pack size
  limit and exclusions, discard refuses a launched dir), end-to-end remote EMT
  optimize through fake ssh with energy equal to a local run, live events
  arrive while running, cancel, unreachable mid-run then recovery without
  failing the job, server restart reattaches, two queues run in parallel
  (local + remote) while one machine's queue stays serial, copy-back threshold
  (a file over the limit is remote-only and streams through `cat`),
  `/api/paths/check` remote, calculators per target, 409 on deleting a busy
  machine.
- Every test asserts a number or invariant. Existing tests keep passing
  unchanged (`pytest tests/webapp -m "not mace"`, then the full unit command
  from AGENTS.md).

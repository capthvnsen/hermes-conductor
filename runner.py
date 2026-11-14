"""Start, read, and stop captain runs. The process table stays in plugin-data."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("hermes_captain.runner")

from .peek import (
    approval_stuck,
    build_card,
    cache_policy,
    error_reason,
    git_stat,
    hold_reply,
    idle_seconds,
    STALLED_AFTER,
    reset_time,
    run_health,
    t3_pending,
    t3_thread,
    thread_failure,
)
from .plan import (
    ALIASES,
    EVENT_KINDS,
    HOST_RE,
    RECIPES,
    PlanError,
    plan_event_wait,
    plan_inspect,
    plan_list,
    plan_orca_read,
    plan_orca_wait,
    plan_read,
    plan_run,
    plan_steer,
    plan_stop,
    plan_tmux_stop,
    plan_wait,
    ssh_argv,
)

MAX_READ = 8000
MAX_RESULT = 1500
T3_POLL = 60  # seconds per T3 wait; the health check runs between waits
ORCA_POLL_MS = 60_000  # one tui-idle wait; a timeout means still working, not finished
# shortcut: a just-opened agent TUI is idle before it takes the prompt, so idle counts only this long after a turn starts
ORCA_SETTLE = 20
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
CARD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
MODES = ("drive", "advise")
MAX_ROUNDS = 5
GIVE_UP_AFTER = 86400
REDELIVER_EVERY = 60  # seconds between tries at a ping Hermes did not accept
# shortcut: claude -p prints nothing until it ends, so silence is normal; 30 minutes instead of 10 before "stalled"
SILENT_STALL = 1800
DOCTOR = ("hermes", "claude", "codex", "grok", "orca", "t3code", *(words[0] for words, _ in RECIPES.values()))
NAMED_AGENTS = frozenset({"hermes", "claude", "codex", "grok", "orca", "t3", *RECIPES})


def default_data_dir() -> Path:
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("conductor")


class Captain:
    def __init__(self, data_dir: Path | None = None, spawn=None, capture=None, diff=None):
        self.data_dir = data_dir or default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.spawn = spawn or _spawn
        self.capture = capture or _capture
        self.diff = diff or git_stat
        self._procs: dict[int, subprocess.Popen] = {}
        self._waits: dict[str, threading.Thread] = {}
        self._tried: dict[str, float] = {}  # when this process last tried to deliver a run's ping
        self.last_peek: dict[str, dict] = {}  # per chat: a peek in one chat never rewrites a reply in another
        # None means no limit (tests, direct use). The plugin always sets both from the owner's settings, empty by default.
        self.allowed_hosts: set[str] | None = None
        self.allowed_commands: set[str] | None = None
        self.notify = None
        self.retry_delay = 5
        self.health_every = 30
        self.events_dir = Path(os.environ.get("CONDUCTOR_EVENTS") or Path.home() / ".conductor" / "events")
        self._kanban_threads: list[threading.Thread] = []
        self._kanban_logged = False
        self._supervise_stop = threading.Event()

    def get_mode(self) -> str:
        """drive or advise. advise unless the user chose drive: no surprise spend."""
        try:
            mode = json.loads((self.data_dir / "mode.json").read_text(encoding="utf-8")).get("mode")
        except (OSError, json.JSONDecodeError, AttributeError):
            return "advise"
        return mode if mode in MODES else "advise"

    def set_mode(self, mode: str) -> str:
        mode = _check_mode(mode)
        (self.data_dir / "mode.json").write_text(json.dumps({"mode": mode}) + "\n", encoding="utf-8")
        return mode

    def _check_host(self, host: str | None) -> None:
        if host and self.allowed_hosts is not None and host not in self.allowed_hosts:
            raise PlanError(
                f"host {host} is not allowed. Only the owner can allow it: "
                "add it to plugins.entries.conductor.settings.hosts in the Hermes config."
            )

    def _check_agent(self, agent: str) -> None:
        name = ALIASES.get((agent or "").strip().lower(), (agent or "").strip().lower())
        if name in NAMED_AGENTS or self.allowed_commands is None or name in self.allowed_commands:
            return
        raise PlanError(
            f"{name or 'that'} is not an agent Conductor knows. Only the owner can allow another command: "
            "add it to plugins.entries.conductor.settings.commands in the Hermes config."
        )

    def doctor(self, host: str | None = None) -> dict:
        self._check_host(host)
        if host:
            if not HOST_RE.fullmatch(host):
                raise PlanError("host must be a plain SSH destination")
            script = 'for c in "$@"; do if command -v "$c" >/dev/null 2>&1; then echo "$c yes"; else echo "$c no"; fi; done'
            result = self.capture(ssh_argv(host, ["bash", "-c", script, "bash", *DOCTOR]), None, 20)
            found = {}
            for line in (result.get("stdout") or "").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    found[parts[0]] = parts[1] == "yes"
            return {"host": host, "ok": result.get("code") == 0, "agents": found, "error": result.get("error")}
        import shutil
        report = {"host": "local", "ok": True, "agents": {name: shutil.which(name) is not None for name in DOCTOR}}
        unsent = sum(1 for record in self._records() if record.get("follow_up") == "undelivered")
        if unsent:  # the usual cause on a gateway; without it Hermes never hears that a run finished
            report["error"] = (
                f"{unsent} finish message(s) could not be delivered to Hermes. "
                "Set plugins.entries.conductor.allow_gateway_injection: true in the Hermes config."
            )
        return report

    def run(self, agent: str, prompt: str, **kwargs) -> dict:
        self._check_agent(agent)
        self._check_host(kwargs.get("host"))
        self._check_host(kwargs.get("orca_host"))
        keep = bool(kwargs.pop("keep", False))
        session_key = str(kwargs.pop("session_key", "") or "")
        mode = kwargs.pop("mode", None)
        mode = _check_mode(mode) if mode else None
        card = str(kwargs.pop("card", "") or "")
        if card and not CARD_RE.fullmatch(card):
            raise PlanError("card must be a plain task id")
        run_id = uuid.uuid4().hex[:12]
        plan = plan_run(agent, prompt, run_id=run_id, **kwargs)
        record = {
            "id": run_id,
            "agent": plan.name or agent,
            "surface": plan.surface,
            "mode": plan.mode,
            "summary": plan.summary,
            "argv0": plan.argv[0],
            "host": kwargs.get("host") or "",
            "cwd": kwargs.get("cwd") or "",
            "keep": keep,
            "last_active": time.time(),
            "events": plan.push,
            "owner": os.getpid(),
        }
        if session_key:
            record["session_key"] = session_key
        if card:
            record["card"] = card
        if mode:
            record["follow_mode"] = mode  # "mode" on the record is already the launch mode
        if plan.mode == "capture":
            result = self.capture(plan.argv, plan.cwd, 90)
            stdout = result.get("stdout") or ""
            record["status"] = "launched" if result.get("code") == 0 else "error"
            record["handle"] = _guess_handle(stdout)
            # the launch reply is a large JSON wrapper: keep it out of the model's context unless it is the error
            record["output"] = "" if result.get("code") == 0 else _clip(result.get("error") or stdout)
            if plan.surface == "orca":
                record["outcome"] = _orca_outcome(stdout)
                environment = (kwargs.get("orca_host") or "").strip()
                if environment:
                    record["environment"] = environment
                # A chat has no documented wait. Leave it launched. Do not call that finished.
                record["watched"] = (
                    record["status"] == "launched"
                    and record["outcome"] == "terminal"
                    and bool(record.get("handle"))
                )
                if record["status"] == "launched" and not record["watched"]:
                    record["summary"] = "orca launch is not watched: it did not open a terminal"
                elif record["watched"]:
                    record["summary"] = "orca terminal launch, waiting with no model"
        else:
            log_path = self.data_dir / f"{run_id}.log"
            pid = self.spawn(plan.argv, plan.cwd, log_path)
            if self.spawn is _spawn:
                proc = getattr(_spawn, "_last", None)
                if proc is not None:
                    self._procs[proc.pid] = proc
            record["status"] = "running"
            record["pid"] = pid
            record["pid_started"] = _proc_start(pid)
            record["log"] = str(log_path)
        self._save(run_id, record)
        self._kanban(record, "comment", f"started: {record['agent']} {record['host'] or 'local'}")
        if record.get("pid") or (record.get("surface") == "t3" and record.get("handle")) or record.get("watched"):
            self.arm_wait(run_id)
        return record

    def read(self, run_id: str) -> dict:
        record = self._load(run_id)
        if record.get("surface") == "t3" and record.get("handle"):
            plan = plan_read(record.get("agent") or "t3", record["handle"], host=record.get("host") or None)
            if plan is not None:
                result = self.capture(plan.argv, None, 30)
                if result.get("code") == 0:  # the answer, never the JSON wrapper: it is 10x the tokens and changes every read
                    record["output"] = _clip(_answer_text(result.get("stdout") or ""))
                else:
                    record["output"] = _clip(result.get("error") or "")
                    record["status"] = "error"
                self._save(run_id, record)
                return record
        log = record.get("log")
        if log and Path(log).exists():
            record["output"] = _clip(Path(log).read_text(encoding="utf-8", errors="replace"))
            pid = record.get("pid")
            record["status"] = "running" if pid and self._running(pid) else "exited"
        return record

    def peek(self, run_id: str, session: str = "") -> dict:
        record = self.read(run_id)
        output = record.get("output") or ""
        alive = record.get("status") in {"running", "launched"}
        diff = ""
        cwd = record.get("cwd") or ""
        host = record.get("host") or ""
        if cwd and host:
            diff = self._remote_status(host, cwd)
        elif cwd:
            diff = self.diff(cwd) or ""
        if record.get("surface") == "t3" and record.get("handle"):
            record["inspect"] = self._inspect(record)
        stuck = t3_pending(record.get("inspect") or "") or approval_stuck(output, alive)
        health, why = self._health(record, output, alive, stuck, t3_thread(record.get("inspect") or ""))
        card = build_card(
            record,
            output=output,
            alive=alive,
            diff=diff,
            previous=record.get("fingerprint"),
            stuck=stuck,
            health=health,
            why=why,
        )
        record["fingerprint"] = card["fingerprint"]
        record["stuck_on_approval"] = card["stuck_on_approval"]
        self._save(run_id, record)
        self.last_peek[session] = {"speak": card["speak"], "run_id": run_id}
        return card

    def supervise(self, interval: float = 2.0) -> bool:
        """Keep waits alive after the turn that started them has exited.

        The lock is held for the life of this process. A short-lived turn worker
        loses the race to the gateway, which is the process that can message the user.
        """
        import fcntl

        if getattr(self, "_supervisor", None) and self._supervisor.is_alive():
            return True
        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.data_dir / ".supervisor.lock"
        lock_path.touch()
        handle = open(lock_path, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._lock_handle = handle

        def loop() -> None:
            while not self._supervise_stop.is_set():
                try:
                    self.resume()
                except Exception:
                    logger.exception("conductor supervise")
                self._supervise_stop.wait(interval)

        thread = threading.Thread(target=loop, name="conductor-supervise", daemon=True)
        thread.start()
        self._supervisor = thread
        return True

    def arm_wait(self, run_id: str) -> dict:
        """Wait for the run with no model. A 15-minute check cannot beat this."""
        record = self._load(run_id)
        if record.get("pinged_at"):
            return {"ok": True, "ping": record.get("ping"), "seconds": 0, "model": False}
        thread = self._waits.get(run_id)
        if thread is not None and thread.is_alive():  # a second wait would ping twice
            return {"ok": True, "waiting": True, "model": False}
        record["wait_started"] = time.time()
        self._save(run_id, record)
        thread = threading.Thread(target=self._wait_loop, args=(run_id,), daemon=True)
        thread.start()
        self._waits[run_id] = thread
        return {"ok": True, "waiting": True, "model": False}

    def watch_running(self, host: str | None = None, session_key: str = "") -> dict:
        """Arm a no-model wait on every running T3 thread. Ping when one finishes."""
        self._check_host(host)
        plan = plan_list(host)
        result = self.capture(plan.argv, None, 30)
        if result.get("code") != 0:
            return {"ok": False, "error": result.get("error") or "could not list threads", "watched": []}
        try:
            data = json.loads(result.get("stdout") or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "thread list was not json", "watched": []}
        threads = (data.get("data") or {}).get("threads") or []
        watching = self._open_handles()
        watched = []
        for thread in threads:
            if thread.get("runStatus") != "running" and thread.get("activityRunStatus") != "running":
                continue
            handle = thread.get("id") or ""
            if not handle or handle in watching:
                continue
            run_id = uuid.uuid4().hex[:12]
            record = {
                "id": run_id,
                "agent": "t3",
                "surface": "t3",
                "handle": handle,
                "host": host or "",
                "title": thread.get("title") or "",
                "status": "running",
                "keep": False,
                "last_active": time.time(),
            }
            if session_key:
                record["session_key"] = session_key
            self._save(run_id, record)
            self.arm_wait(run_id)
            watched.append({"run_id": run_id, "title": record["title"]})
        return {"ok": True, "watched": watched, "model": False}

    def _open_handles(self) -> set[str]:
        return {r["handle"] for r in self._records() if r.get("handle") and not r.get("pinged_at")}

    def _records(self) -> list[dict]:
        records = []
        for path in sorted(self.data_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and RUN_ID_RE.fullmatch(str(record.get("id") or "")):
                records.append(record)
        return records

    def _wait_t3(self, record: dict) -> str:
        host = record.get("host") or None
        fails = 0
        deadline = time.time() + 4 * 3600
        while time.time() < deadline:
            plan = plan_wait(record["handle"], host=host, timeout=T3_POLL)
            result = self.capture(plan.argv, None, T3_POLL + 20)
            outcome = _wait_outcome(result.get("stdout") or "") or _wait_outcome(result.get("stderr") or "")
            if outcome == "completed":
                return "done"
            if outcome == "needs-attention":
                if record.get("ping") != "needs-attention":
                    return "needs-attention"
                time.sleep(self.retry_delay)  # Hermes was told once: wait quietly for the approval, the finish is still owed
                continue
            if outcome in ("error", "interrupted"):
                return "errored"
            if outcome == "timeout" or result.get("code") in (6, 124):
                fails = 0
                if self._tick(record["id"]) == "errored":
                    return "errored"
                continue
            if result.get("code") == 0 and not outcome:
                return "done"
            if outcome and self._tick(record["id"]) == "errored":  # a wait answer we do not know may be a failed run
                return "errored"
            fails += 1  # ssh to the other machine can blip; three in a row is a real failure
            if fails >= 3:
                return "wait failed"
            time.sleep(self.retry_delay)
        return "wait failed"

    def _wait_orca(self, record: dict) -> str:
        """Wait on the terminal Orca opened. tui-idle is the turn settling. A block is stuck, not finished.
        A timeout means the TUI is still busy, so keep waiting. There is no chat wait."""
        host = record.get("host") or None
        environment = record.get("environment") or None
        fails = 0
        for _ in range(4 * 3600 * 1000 // ORCA_POLL_MS):
            try:
                plan = plan_orca_wait(
                    record["handle"],
                    condition="tui-idle",
                    timeout_ms=ORCA_POLL_MS,
                    host=host,
                    environment=environment,
                )
            except PlanError:
                return "wait failed"
            result = self.capture(plan.argv, None, ORCA_POLL_MS // 1000 + 20)
            wait = _orca_wait(result.get("stdout") or "") or _orca_wait(result.get("stderr") or "")
            idle = wait.get("satisfied") is True and not wait.get("blockedReason")
            if idle and wait.get("status") != "exited" and time.time() - (self._load(record["id"]).get("last_active") or 0) < ORCA_SETTLE:
                time.sleep(self.retry_delay)
                continue
            if wait.get("status") == "exited" or idle:
                code = wait.get("exitCode")
                if isinstance(code, int) and code:
                    saved = self._load(record["id"])
                    saved["exit_code"] = code
                    self._save(record["id"], saved)
                return "done"
            if wait.get("blockedReason"):
                return "needs-attention"
            # exit 1 is also a stale handle or a closed Orca: only a timeout means still working
            if wait.get("satisfied") is False or result.get("code") == 124 or "timeout" in (_orca_error(result.get("stdout") or ""), _orca_error(result.get("stderr") or "")):
                fails = 0
                if self._tick(record["id"]) == "errored":
                    return "errored"
                continue
            fails += 1
            if fails >= 3:
                return "wait failed"
            time.sleep(self.retry_delay)
        return "wait failed"

    def wait_done(self, run_id: str, timeout: float = 30) -> dict:
        thread = self._waits.get(run_id)
        if thread is not None:
            thread.join(timeout)
        record = self._load(run_id)
        started = record.get("wait_started") or record.get("pinged_at") or time.time()
        pinged = record.get("pinged_at")
        return {
            "ok": bool(pinged),
            "ping": record.get("ping"),
            "seconds": round((pinged - started), 3) if pinged else None,
            "model": False,
            "status": record.get("status"),
        }

    def _wait_loop(self, run_id: str) -> None:
        try:
            while self._wait_once(run_id):
                pass
        except PlanError:
            return  # the record is gone
        except Exception:
            logger.exception("conductor wait %s", run_id)  # supervise() arms it again

    def _wait_once(self, run_id: str) -> bool:
        """One wait and one ping. True when the agent asked for input but is still running: its finish is still owed."""
        record = self._load(run_id)
        ping = "done"
        code, event = None, None
        if record.get("surface") == "t3" and record.get("handle"):
            ping = self._wait_t3(record)
        elif record.get("surface") == "orca" and record.get("outcome") == "terminal" and record.get("handle"):
            ping = self._wait_orca(record)
        elif record.get("pid"):
            code, event = self._wait_pid(run_id, record["pid"])
            if event and event.get("kind") == "needs-input":
                ping = "needs-attention"
        else:
            ping = "no process to wait on"
        fresh = self._load(run_id)
        if code and fresh.get("status") != "stopped":  # our own SIGTERM is not a crash
            fresh["exit_code"] = code
        if event:
            fresh["event"] = event["kind"]
            fresh["event_message"] = str(event.get("message") or "")[-MAX_RESULT:]
        if ping in ("done", "errored") and fresh.get("status") != "stopped":
            fresh["status"] = "exited"
        fresh["ping"] = ping
        fresh["pinged_at"] = fresh["last_active"] = time.time()
        fresh["message"] = self._finish_message(fresh, ping)
        self._save(run_id, fresh)
        self._deliver(run_id, fresh)
        if ping != "needs-attention":
            return False
        return bool(record.get("surface") == "t3" and record.get("handle")) or bool(record.get("pid") and self._running(record["pid"]))

    def _wait_pid(self, run_id: str, pid: int) -> tuple[int | None, dict | None]:
        """Block until the process exits or the agent's own hook reports, checking health on the way.
        Returns the exit code (when we are its parent) and the event, if one woke us."""
        record = self._load(run_id)
        pushed = record.get("events", "polling") != "polling"
        found: list[dict] = []
        stop = threading.Event()
        if pushed:
            threading.Thread(target=self._pull_event, args=(record, found, stop), daemon=True).start()
        last = time.time()
        try:
            while self._running(pid) and not found:
                time.sleep(0.2)
                if time.time() - last >= self.health_every:
                    last = time.time()
                    self._tick(run_id)
        finally:
            stop.set()
        if pushed and not found:
            self._take_event(record)  # the process won the race: still clear the file the hook left
        proc = self._procs.get(int(pid))
        return (proc.returncode if proc is not None else None), (found[0] if found else None)

    def _pull_event(self, record: dict, found: list, stop: threading.Event) -> None:
        remote = bool(record.get("host"))
        while not stop.is_set() and not found:
            try:
                event = self._take_event(record, 60 if remote else 0)
            except Exception:
                logger.exception("conductor event wait")
                event = None
            if event:
                found.append(event)
                return
            stop.wait(1 if remote else 0.2)

    def _take_event(self, record: dict, seconds: int = 0) -> dict | None:
        """Look for the run's event file, wait up to `seconds` if it is on another machine. Reads it, deletes it.
        A broken file is deleted and ignored, so polling carries on."""
        run_id = record["id"]
        if record.get("host"):
            got = self.capture(plan_event_wait(run_id, record["host"], seconds).argv, None, seconds + 15)
            text = got.get("stdout") or ""
        else:
            text = ""
            for kind in EVENT_KINDS:
                try:
                    text = (self.events_dir / f"{run_id}.{kind}.json").read_text(encoding="utf-8")
                    break
                except OSError:
                    continue
            for path in self.events_dir.glob(f"{run_id}.*.json"):
                path.unlink(missing_ok=True)
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return None
        return event if isinstance(event, dict) and event.get("kind") in EVENT_KINDS else None

    def _health(self, record: dict, output: str, alive: bool, stuck: bool, thread: dict) -> tuple[str, str]:
        state = thread.get("runStatus")
        reason = thread_failure(thread)
        if state:  # a T3 thread knows better than the record whether it is still going
            alive = state not in ("idle", "completed") or stuck
        if not alive and not reason:
            reason = error_reason(record.get("exit_code"), output)
        return run_health(
            alive=alive, idle=idle_seconds(record, thread), stuck=stuck, reason=reason,
            stalled_after=SILENT_STALL if record.get("events") == "push" else STALLED_AFTER,
        )

    def _tick(self, run_id: str) -> str:
        """One no-model health check inside the wait. Pings once per stall or approval."""
        record = self._load(run_id)
        t3 = record.get("surface") == "t3" and record.get("handle")
        inspect = self._inspect(record) if t3 else ""
        output = "" if t3 else self.read(run_id).get("output") or ""
        stuck = t3_pending(inspect) or approval_stuck(output, True)
        word, why = self._health(record, output, True, stuck, t3_thread(inspect))
        alerted = record.get("alerted") or []
        if word in ("stalled", "needs-approval") and word not in alerted:
            record["alerted"] = [*alerted, word]
            self._save(run_id, record)
            name = record.get("title") or run_id
            where = f" on {record['host']}" if record.get("host") else ""
            message = (
                f"Conductor: {name}{where} is {word}: {why}. Tell the user this now, in one line. "
                f"Check it with conductor_peek {run_id}. Do not steer it and do not start another run."
            )
            if word == "needs-approval":
                self._kanban(record, "block", why, "--kind", "needs_input")
            else:
                self._kanban(record, "comment", f"{word}: {why}")
            try:  # shortcut: a ping that fails to send is not retried; the finish ping still is
                if self.notify:
                    self.notify(run_id, {**record, "message": message})
            except Exception:
                logger.exception("conductor alert")
        elif word == "working" and alerted:
            record["alerted"] = []
            self._save(run_id, record)
        return word

    def _deliver(self, run_id: str, record: dict) -> None:
        try:
            sent = bool(self.notify and self.notify(run_id, record))
        except Exception:
            sent = False
        record["follow_up"] = "delivered" if sent else "undelivered"
        self._tried[run_id] = time.time()
        self._save(run_id, record)

    def resume(self) -> None:
        """After a restart: re-send missed messages and re-arm T3 waits."""
        for record in self._records():
            run_id = record["id"]
            owner = record.get("owner")
            if owner and owner != os.getpid() and _alive(owner):
                continue  # the process that started it is still up: it waits and it pings, we would only ping twice
            if record.get("follow_up") == "undelivered" and record.get("message"):
                if time.time() - (record.get("pinged_at") or time.time()) > GIVE_UP_AFTER:
                    continue  # a day-old ping has no chat left to reach; retrying only fills the log
                if time.time() - self._tried.get(run_id, 0) >= REDELIVER_EVERY:
                    self._deliver(run_id, record)
            elif record.get("resume_at") and not record.get("resumed_at") and time.time() >= record["resume_at"]:
                self._nudge(run_id, record)
            elif record.get("pinged_at") or (run_id in self._waits and self._waits[run_id].is_alive()):
                continue
            elif record.get("surface") == "orca" and record.get("outcome") != "terminal":
                continue  # a chat launch has no wait; do not pretend it finished
            elif self.allowed_hosts is not None and {record.get("host"), record.get("environment")} - {None, ""} - self.allowed_hosts:
                continue  # its host is no longer on the owner's list: do not reach it
            elif record.get("handle"):
                self.arm_wait(run_id)
            elif record.get("pid"):
                started = _proc_start(record["pid"])
                if started and started == record.get("pid_started"):
                    self.arm_wait(run_id)
                else:  # a pid alone can belong to a stranger now
                    record["ping"] = "wait lost"
                    record["pinged_at"] = time.time()
                    record["message"] = self._finish_message(record, "wait lost")
                    self._save(run_id, record)
                    self._deliver(run_id, record)

    def _nudge(self, run_id: str, record: dict) -> None:
        """Once, after a usage limit should have reset: tell Hermes. Hermes sends the continue, never this plugin."""
        name = record.get("title") or run_id
        if record.get("surface") == "t3" and record.get("handle"):
            go = f"conductor_steer on run {run_id}"
        else:
            go = f"conductor_run (agent {record.get('agent')}, continue true, cwd {record.get('cwd') or 'the same directory'})"
        go += ": 'Continue where you left off.'"
        head = f"Conductor: {name} hit a usage limit and it should have reset."
        if (record.get("rounds") or 0) >= MAX_ROUNDS:
            text = f"{head} Run {run_id} has used all {MAX_ROUNDS} follow-ups. Tell the user and wait. Do not call conductor_steer or conductor_run."
        elif (record.get("follow_mode") or self.get_mode()) == "advise":
            text = f"{head} Ask the user first whether to resume it with {go}"
        else:
            text = f"{head} Resume it with {go}"
        try:
            sent = bool(self.notify and self.notify(run_id, {**record, "message": text}))
        except Exception:
            sent = False
        if sent:  # an unsent nudge is tried again on the next pass
            record["resumed_at"] = time.time()
            self._save(run_id, record)

    def _kanban(self, record: dict, verb: str, text: str, *flags: str) -> None:
        """Mirror an event onto the Hermes board. Off the wait's thread, never raises, never completes a card."""
        card = record.get("card")
        if not card:
            return
        argv = ["hermes", "kanban", verb, *flags, card, f"Conductor: {text}"]

        def call() -> None:
            try:
                got = self.capture(argv, None, 20)
                error = "" if got.get("code") == 0 else (got.get("error") or f"exit {got.get('code')}")
            except Exception as exc:
                error = str(exc) or type(exc).__name__
            if error and not self._kanban_logged:
                self._kanban_logged = True
                logger.warning("conductor kanban %s failed: %s", verb, error)

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        self._kanban_threads = [t for t in self._kanban_threads if t.is_alive()] + [thread]

    def _finish_message(self, record: dict, ping: str) -> str:
        name = record.get("title") or record.get("id")
        where = f" on {record['host']}" if record.get("host") else ""
        run_id = record.get("id")
        if ping not in ("done", "needs-attention", "errored"):
            return (
                f"Conductor: lost track of {name}{where} ({ping}). It may still be running. "
                f"Tell the user this now. Check it once with conductor_peek {run_id}. Do not start another run."
            )
        if record.get("status") == "stopped":
            return f"Conductor: {name}{where} was stopped (run {run_id}). Tell the user in one line. Do not start another run unless the user says so."
        result, pending, failure, lasterr = "", False, "", ""
        if record.get("surface") == "t3" and record.get("handle"):
            host = record.get("host") or None
            read = plan_read(record.get("agent") or "t3", record["handle"], host=host)
            if read is not None:
                got = self.capture(read.argv, None, 30)
                result = _answer_text(got.get("stdout") or "") if got.get("code") == 0 else ""
            inspect = self._inspect(record)
            pending = t3_pending(inspect)
            thread = t3_thread(inspect)
            lasterr = str(thread.get("lastError") or "")
            failure = thread_failure(thread) or ("the thread run failed" if ping == "errored" else "")
        elif record.get("surface") == "orca" and record.get("outcome") == "terminal" and record.get("handle"):
            read = plan_orca_read(
                record["handle"],
                host=record.get("host") or None,
                environment=record.get("environment") or None,
            )
            got = self.capture(read.argv, None, 30)
            result = _orca_tail(got.get("stdout") or "") if got.get("code") == 0 else ""
        elif record.get("log") and Path(record["log"]).exists():
            result = Path(record["log"]).read_text(encoding="utf-8", errors="replace").strip()
        said = record.get("event_message") or ""
        if said:  # the agent's own hook captured its answer; the log also holds its warnings
            result = said
        if len(result) > MAX_RESULT:
            result = f"[first {len(result) - MAX_RESULT} characters cut]\n{result[-MAX_RESULT:]}"
        result = result or "(the agent returned no answer text)"
        if pending or ping == "needs-attention":
            self._kanban(record, "block", "waiting on an approval; approve it on that machine", "--kind", "needs_input")
            return (
                f"Conductor: {name}{where} is waiting on you, probably an approval. "
                f"Tell the user now that it needs their approval on that machine. What it last said:\n{result}\n"
                "Do not steer it and do not start another run to get past the approval."
            )
        reason = failure or error_reason(record.get("exit_code"), result)
        if reason:
            again = ""
            if any(error_reason(None, t) == "usage limit reached" for t in (lasterr, result)):
                at = reset_time(lasterr) or reset_time(result)
                if at:
                    record["resume_at"] = at
                    again = f" It should reset around {time.strftime('%H:%M', time.localtime(at))}."
                else:
                    again = " Reset time unknown."
            self._kanban(record, "block", reason)
            return (
                f"Conductor: {name}{where} errored: {reason} (run {run_id}).{again} "
                f"Tell the user now, in one line. What it last said:\n{result}\n"
                "Do not start another run to retry unless the user says so."
            )
        self._kanban(record, "comment", result)
        mode = record.get("follow_mode") or self.get_mode()
        rounds = record.get("rounds") or 0
        if mode == "advise":
            then = (
                "Then: if the user's goal is met, stop. If it is not, recommend one next step in one line and wait for the user. "
                "Do not call conductor_steer or conductor_run."
            )
        elif rounds >= MAX_ROUNDS:
            then = (
                f"Then: stop. This run has used all {MAX_ROUNDS} follow-ups. Tell the user where it stands and wait. "
                "Do not call conductor_steer or conductor_run."
            )
        else:
            then = (
                f"Then: if the user's goal is met, stop. If it is not, send one follow-up with conductor_steer on run {run_id} "
                f"(round {rounds + 1} of {MAX_ROUNDS}). Stop and tell the user at once if the run is stalled, errored, "
                "needs approval, or lost. Never start another run to get past an approval."
            )
        return (
            f"Conductor: {name}{where} finished (run {run_id}). "
            f"Message the user now with this result, in your own words:\n{result}\n{then}"
        )

    def _inspect(self, record: dict) -> str:
        plan = plan_inspect(record["handle"], host=record.get("host") or None)
        result = self.capture(plan.argv, None, 20)
        return result.get("stdout") or ""

    def _remote_status(self, host: str, cwd: str) -> str:
        argv = ssh_argv(host, ["git", "-C", cwd, "status", "--short"])
        result = self.capture(argv, None, 15)
        return (result.get("stdout") or "").strip()[:500]

    def _running(self, pid: int) -> bool:
        proc = self._procs.get(int(pid))
        if proc is not None:
            return proc.poll() is None
        return _alive(pid)

    def hold_status_reply(self, reply: str, session: str = "") -> str | None:
        last = self.last_peek.pop(session, None)  # one reply only: a later answer is not a status line
        return hold_reply(last, reply)

    def steer(self, run_id: str, prompt: str) -> dict:
        record = self._load(run_id)
        host = record.get("host") or None
        plan = plan_steer(
            record.get("agent") or "", record.get("handle") or "", prompt,
            host=host, environment=record.get("environment") or None,
        )
        policy = {}
        if plan.surface == "t3":
            probe = self.capture(plan_inspect(record["handle"], host=host).argv, None, 20)
            if probe.get("code") != 0:
                return {"ok": False, "output": "could not check for a pending approval, so nothing was sent"}
            if t3_pending(probe.get("stdout") or ""):
                return {"ok": False, "output": "stuck on an approval; approve it on that machine first. Nothing was sent"}
            thread = t3_thread(probe.get("stdout") or "")
            policy = cache_policy(idle_seconds(record, thread, use_last_active=True), thread)
        elif not record.get("watched"):
            return {"ok": False, "output": "this Orca launch has no terminal to send to. Nothing was sent"}
        result = self.capture(plan.argv, None, 60)
        ok = result.get("code") == 0
        if ok:
            record = self._load(run_id)
            record["last_active"] = time.time()
            record["rounds"] = (record.get("rounds") or 0) + 1
            waiting = self._waits.get(run_id)
            if not (waiting is not None and waiting.is_alive()):
                # the last turn already pinged; this new turn must report back too
                for key in ("pinged_at", "ping", "follow_up", "message", "alerted", "resume_at", "resumed_at"):
                    record.pop(key, None)
                record["status"] = "launched"
            self._save(run_id, record)
            self.arm_wait(run_id)
        return {"ok": ok, "output": "sent" if ok else _clip(result.get("error") or result.get("stdout") or ""), **policy}

    def stop(self, run_id: str) -> dict:
        record = self._load(run_id)
        host = record.get("host") or None
        pid = record.get("pid")
        remote = plan_stop(record.get("agent") or "", record.get("handle") or "", host=host, environment=record.get("environment") or None)
        if remote is None and host and pid:
            remote = plan_tmux_stop(run_id, host)
        if remote is not None:
            result = self.capture(remote.argv, None, 30)
            ok = result.get("code") == 0
            if ok:  # a stop that did not land (the turn had already ended) changes nothing
                record = self._load(run_id)
                record["status"] = "stopped"
                self._save(run_id, record)
            return {"ok": ok, "output": "stopped" if ok else _clip(result.get("error") or result.get("stdout") or "")}
        if not pid or host:
            return {"ok": False, "output": "nothing to stop: this run has no process, thread, or terminal"}
        if record.get("status") != "running":
            return {"ok": False, "output": "nothing to stop: this run is no longer running"}
        started = record.get("pid_started")
        if not started or _proc_start(pid) != started:
            return {"ok": False, "output": "process identity could not be verified; nothing signaled"}
        try:
            if os.getpgid(int(pid)) != int(pid):
                return {"ok": False, "output": "process group could not be verified; nothing signaled"}
            if _proc_start(pid) != started:  # it may have exited while its group was checked
                return {"ok": False, "output": "process identity changed; nothing signaled"}
            os.killpg(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            return {"ok": False, "output": "process has exited; nothing signaled"}
        except PermissionError:
            return {"ok": False, "output": "permission denied signaling the process group; nothing signaled"}
        record["status"] = "stopped"
        self._save(run_id, record)
        return {"ok": True, "output": f"signaled {pid}"}

    def _path(self, run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(run_id or ""):
            raise PlanError("run id is not a plain id")
        return self.data_dir / f"{run_id}.json"

    def _save(self, run_id: str, record: dict) -> None:
        path = self._path(run_id)  # write beside it, then swap: a reader never sees half a record
        tmp = path.with_name(f"{run_id}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _load(self, run_id: str) -> dict:
        path = self._path(run_id)
        if not path.exists():
            raise PlanError(f"no run {run_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        # a saved run reaches its host on read, peek, wait, steer and stop: the owner's list applies to those too
        self._check_host(record.get("host"))
        self._check_host(record.get("environment"))
        return record


def _check_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise PlanError("mode must be drive or advise")
    return mode


def _wait_outcome(stdout: str) -> str:
    try:
        data = json.loads(stdout or "")
    except json.JSONDecodeError:
        return ""
    found = []

    def walk(node):
        if isinstance(node, dict):
            wait = node.get("wait")
            if isinstance(wait, dict) and isinstance(wait.get("outcome"), str):
                found.append(wait["outcome"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    if found:
        return found[0]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict) and error.get("code") == "THREAD_WAIT_TIMEOUT":
        return "timeout"
    return ""


def _answer_text(stdout: str) -> str:
    """The assistant's final answer from `t3code threads read`, never the JSON wrapper."""
    try:
        data = json.loads(stdout or "")
    except json.JSONDecodeError:
        return (stdout or "").strip()
    inner = data.get("data") if isinstance(data, dict) else None
    thread = inner.get("thread") if isinstance(inner, dict) else None
    if not isinstance(thread, dict):
        return ""
    messages = [m for m in thread.get("messages") or [] if isinstance(m, dict) and m.get("role") == "assistant" and not m.get("streaming")]
    turns = [t for t in thread.get("turns") or [] if isinstance(t, dict)]
    final = turns[-1].get("finalMessageId") if turns else None
    for message in messages:
        if message.get("id") == final and isinstance(message.get("text"), str):
            return message["text"].strip()
    texts = [m["text"].strip() for m in messages if isinstance(m.get("text"), str) and m["text"].strip()]
    return texts[-1] if texts else ""


def _clip(text: str) -> str:
    text = text or ""
    if len(text) <= MAX_READ:
        return text
    return text[-MAX_READ:]


def _orca_outcome(output: str) -> str:
    """`terminal` or `structured` (a chat) from `orca agent launch --json`. Anything else is not watched."""
    try:
        data = json.loads(output or "")
    except json.JSONDecodeError:
        return ""
    result = data.get("result") if isinstance(data, dict) else None
    outcome = result.get("outcome") if isinstance(result, dict) else None
    kind = outcome.get("kind") if isinstance(outcome, dict) else None
    if isinstance(kind, str) and re.fullmatch(r"[a-z][a-z-]{0,40}", kind):
        return kind
    return ""


def _orca_wait(stdout: str) -> dict:
    try:
        data = json.loads(stdout or "")
    except json.JSONDecodeError:
        return {}
    result = data.get("result") if isinstance(data, dict) else None
    wait = result.get("wait") if isinstance(result, dict) else None
    return wait if isinstance(wait, dict) else {}


def _orca_error(stdout: str) -> str:
    try:
        data = json.loads(stdout or "")
    except json.JSONDecodeError:
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    return str(error.get("code") or "") if isinstance(error, dict) else ""


def _orca_tail(stdout: str) -> str:
    """Answer text from `orca terminal read --json`. The tail, never the JSON wrapper."""
    try:
        data = json.loads(stdout or "")
    except json.JSONDecodeError:
        return (stdout or "").strip()
    result = data.get("result") if isinstance(data, dict) else None
    terminal = result.get("terminal") if isinstance(result, dict) else None
    if not isinstance(terminal, dict):
        return ""
    tail = terminal.get("tail")
    if isinstance(tail, list):
        return "\n".join(str(line) for line in tail if str(line).strip()).strip()
    if isinstance(tail, str):
        return tail.strip()
    return ""


def _guess_handle(output: str) -> str:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    nodes = [data]
    inner = data.get("data")
    if isinstance(inner, dict):
        nodes.append(inner)
        if isinstance(inner.get("thread"), dict):
            nodes.insert(0, inner["thread"])
    result = data.get("result")
    if isinstance(result, dict):
        outcome = result.get("outcome")
        if isinstance(outcome, dict) and isinstance(outcome.get("handle"), str):
            return outcome["handle"]
    if isinstance(data.get("thread"), dict):
        nodes.insert(0, data["thread"])
    for node in nodes:
        for key in ("threadId", "thread_id", "id"):
            value = node.get(key)
            if isinstance(value, str) and value and key != "id" or (key == "id" and "title" in node):
                if isinstance(value, str) and value:
                    return value
    return ""


def _proc_start(pid: int) -> str:
    """Start time of a pid. Same pid with a different start time is a different process."""
    try:
        done = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(int(pid))], env=_child_env(["ps"]),
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return ""
    return (done.stdout or "").strip()


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


BASE_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TZ",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "CONDUCTOR_RUN", "CONDUCTOR_EVENTS",
)
# What each command needs beyond the base. A name ending in * is a prefix.
AGENT_ENV = {
    "ssh": ("SSH_AUTH_SOCK",),
    "hermes": ("HERMES_HOME",),
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN"),
    "codex": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_HOME"),
    "grok": ("XAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "qwen": ("DASHSCOPE_API_KEY",),
    "copilot": ("GH_TOKEN", "GITHUB_TOKEN"),
    "amp": ("AMP_API_KEY",),
    "droid": ("FACTORY_API_KEY",),
    "cursor-agent": ("CURSOR_API_KEY",),
    "orca": ("ORCA_*",),
    "t3code": ("T3CODE_*",),
}


def _child_env(argv: list[str]) -> dict[str, str]:
    """A child gets the base and what its own command needs, never every key Hermes loaded."""
    words = list(argv)
    if words[:1] == ["env"]:  # `env CONDUCTOR_RUN=id claude ...`: the agent is the word after the assignments
        words = [word for word in words[1:] if "=" not in word][:1]
    names = BASE_ENV + AGENT_ENV.get(Path(words[0]).name if words else "", ())
    return {
        key: value for key, value in os.environ.items()
        if key in names or any(name.endswith("*") and key.startswith(name[:-1]) for name in names)
    }


def _spawn(argv: list[str], cwd: str | None, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:  # the child keeps its own copy of the handle
        proc = subprocess.Popen(
            argv,
            cwd=cwd or None,
            env=_child_env(argv),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _spawn._last = proc  # type: ignore[attr-defined]
    return proc.pid


def _capture(argv: list[str], cwd: str | None, timeout: int) -> dict:
    try:
        done = subprocess.run(
            argv,
            cwd=cwd or None,
            env=_child_env(argv),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"code": 124, "stdout": exc.stdout or "", "error": "timed out"}
    except FileNotFoundError as exc:
        return {"code": 127, "stdout": "", "error": f"not found: {exc.filename}"}
    except OSError as exc:
        return {"code": 126, "stdout": "", "error": str(exc)}
    stdout = done.stdout or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    stderr = done.stderr or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return {"code": done.returncode, "stdout": stdout, "stderr": stderr, "error": stderr if done.returncode else ""}

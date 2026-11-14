"""Captain plans stay on verified commands and never add a bypass flag."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import importlib
import importlib.util
import types

pkg = types.ModuleType("hermes_captain")
pkg.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
pkg.__package__ = "hermes_captain"
sys.modules["hermes_captain"] = pkg
importlib.import_module("hermes_captain.plan")
importlib.import_module("hermes_captain.runner")

from hermes_captain.plan import PlanError, plan_run, plan_steer, plan_wait  # noqa: E402
from hermes_captain.runner import Captain  # noqa: E402
import hermes_captain.runner as _runner  # noqa: E402

_runner.ORCA_SETTLE = 0  # the grace after an Orca turn starts; one test below turns it on

BANNED = (
    "--dangerously-skip-permissions",
    "--dangerously-bypass-approvals-and-sandbox",
    "--yolo",
    "full-access",
)


def _flags(argv: list[str]) -> str:
    return " ".join(argv)


def test_claude_is_print_mode_without_bypass():
    plan = plan_run("claude", "fix the test")
    assert plan.argv[:2] == ["claude", "-p"]
    assert plan.mode == "background"
    assert not any(flag in _flags(plan.argv) for flag in BANNED)


def test_codex_exec_has_no_bypass():
    plan = plan_run("codex", "fix the test")
    assert plan.argv[:2] == ["codex", "exec"]
    assert not any(flag in _flags(plan.argv) for flag in BANNED)


def test_orca_launch_matches_the_documented_command():
    plan = plan_run("orca", "write the design doc", worker="codex", orca_host="local")
    assert plan.argv == [
        "orca", "agent", "launch", "--agent", "codex",
        "--prompt", "write the design doc", "--surface", "terminal", "--json", "--environment", "local",
    ]
    assert plan.mode == "capture"


def test_t3_handover_requires_approval():
    plan = plan_run("t3", "review the diff")
    assert "--permission" in plan.argv
    assert plan.argv[plan.argv.index("--permission") + 1] == "approval-required"
    assert "--provider" not in plan.argv
    claude = plan_run("t3", "review the diff", provider="claudeAgent")
    assert claude.argv[claude.argv.index("--provider") + 1] == "claudeAgent"
    assert "full-access" not in plan.argv
    assert plan.mode == "capture"


def test_t3_handover_passes_a_plain_cursor_model():
    plan = plan_run("t3", "review the diff", provider="cursor", model="grok-4.7")
    assert plan.argv[plan.argv.index("--provider") + 1] == "cursor"
    assert plan.argv[plan.argv.index("--model") + 1] == "grok-4.7"
    assert plan.argv[plan.argv.index("--permission") + 1] == "approval-required"
    assert "--model" not in plan_run("t3", "review the diff").argv
    try:
        plan_run("t3", "review the diff", provider="cursor", model="grok 4.7")
    except PlanError as exc:
        assert "plain" in str(exc)
    else:
        raise AssertionError("a model that is not a plain id was accepted")


def test_remote_host_is_one_quoted_command():
    import shlex
    plan = plan_run("claude", "fix it", host="dev@203.0.113.7", cwd="/Users/dev/app")
    assert plan.argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert plan.argv[5] == "dev@203.0.113.7"
    remote = shlex.split(plan.argv[6])
    assert remote[:2] == ["bash", "-c"]
    assert "do" not in remote[2].split() or "for " not in remote[2]
    assert "/Users/dev/app" in remote
    assert "fix it" in remote
    assert "cd /Users" not in plan.argv[4]


def test_bad_host_is_refused():
    try:
        plan_run("claude", "fix it", host="alex@mac;rm -rf /")
    except PlanError as exc:
        assert "host" in str(exc)
    else:
        raise AssertionError("bad host was accepted")


def test_t3_steer_uses_if_busy_steer():
    plan = plan_steer("t3", "thread_123", "try the other file")
    assert plan.argv[plan.argv.index("--if-busy") + 1] == "steer"
    assert plan.argv[plan.argv.index("--thread") + 1] == "thread_123"


def test_terminal_harnesses_do_not_skip_approval():
    kimi = plan_run("kimi", "fix the test")
    assert kimi.argv[:2] == ["kimi", "-p"]
    assert "--yolo" not in kimi.argv and "--auto" not in kimi.argv
    code = plan_run("commandcode", "fix the test")
    assert code.argv[:2] == ["commandcode", "-p"]
    assert "--trust" not in code.argv
    pi = plan_run("pi", "fix the test")
    assert pi.argv[:2] == ["pi", "--print"]
    orca = plan_run("orca", "ping", worktree="path:/Users/dev/Projects/app")
    assert orca.argv[orca.argv.index("--worktree") + 1] == "path:/Users/dev/Projects/app"
    assert "--host" not in orca.argv
    generic = plan_run("worker", "fix the test", prompt_flag="-p")
    assert generic.argv == ["worker", "-p", "fix the test"]
    try:
        plan_run("worker", "fix the test", prompt_flag="--yolo")
    except PlanError:
        return
    raise AssertionError("unknown flag was accepted")
    try:
        plan_steer("claude", "abc", "more")
    except PlanError:
        return
    raise AssertionError("claude steer should be refused")


def test_run_does_not_spawn_when_the_launcher_returns():
    import tempfile
    seen = {}

    def capture(argv, cwd, timeout):
        seen["argv"] = argv
        seen["timeout"] = timeout
        return {"code": 0, "stdout": json.dumps({"threadId": "th_1"}), "error": ""}

    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(
            Path(tmp),
            spawn=lambda *a: (_ for _ in ()).throw(AssertionError("spawned")),
            capture=capture,
        )
        record = captain.run("t3", "review the diff")
        captain.wait_done(record["id"], timeout=8)
    assert record["handle"] == "th_1"
    assert record["status"] == "launched"
    assert seen["argv"][0] == "t3code"
    assert "full-access" not in seen["argv"]


def test_second_peek_stays_quiet_and_hides_the_log():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "run.log"
        log.write_text("still compiling the widget factory\n", encoding="utf-8")
        record = {
            "id": "abc",
            "agent": "claude",
            "status": "running",
            "pid": 1,
            "log": str(log),
            "output": "",
        }
        (root / "abc.json").write_text(json.dumps(record), encoding="utf-8")
        captain = Captain(root, spawn=lambda *a: 1, capture=lambda *a: {"code": 0, "stdout": ""})
        # pid 1 is alive on this machine (launch). Force the status the card cares about.
        first = captain.peek("abc")
        second = captain.peek("abc")
    assert first["speak"] is True
    assert "widget factory" not in json.dumps(first)
    assert second["speak"] is False
    assert second["note"] == "No change."
    assert "output" not in second


def test_changed_peek_shows_diff_and_stuck_not_the_log():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "run.log"
        log.write_text("Do you want to proceed?\nsecret log line\n", encoding="utf-8")
        record = {
            "id": "abc",
            "agent": "claude",
            "status": "running",
            "pid": 1,
            "log": str(log),
            "cwd": "/tmp/does-not-matter",
        }
        (root / "abc.json").write_text(json.dumps(record), encoding="utf-8")
        captain = Captain(
            root,
            spawn=lambda *a: 1,
            capture=lambda *a: {"code": 0, "stdout": ""},
            diff=lambda cwd: "widget.py | 2 +-",
        )
        card = captain.peek("abc")
    assert card["speak"] is True
    assert card["stuck_on_approval"] is True
    assert card["note"] == "Stuck on approval."
    assert card["diff"] == "widget.py | 2 +-"
    assert "secret log line" not in json.dumps(card)


def test_git_stat_is_a_real_diff_not_a_shell():
    import subprocess
    import tempfile
    from hermes_captain.peek import git_stat
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "captain@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Captain"], cwd=root, check=True)
        (root / "widget.py").write_text("print(1)\n", encoding="utf-8")
        subprocess.run(["git", "add", "widget.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)
        (root / "widget.py").write_text("print(1)\nprint(2)\n", encoding="utf-8")
        stat = git_stat(str(root))
    assert "widget.py" in stat
    assert "??" in stat or "M" in stat


def test_handle_comes_from_the_nested_thread():
    from hermes_captain.runner import _guess_handle
    payload = {
        "ok": True,
        "data": {
            "project": {"id": "project-1", "title": "hermes-captain"},
            "thread": {"id": "68cd76c8-61b2-49c9-a608-d0f716d6c63d", "title": "review"},
        },
    }
    assert _guess_handle(json.dumps(payload)) == "68cd76c8-61b2-49c9-a608-d0f716d6c63d"
    from hermes_captain.peek import hold_reply
    assert hold_reply({"speak": False}, "Still running, nothing new.") == "No change."
    assert hold_reply({"speak": True}, "Still running, nothing new.") is None
    assert hold_reply({"speak": False}, "The widget test failed on line 12 because the fixture was empty.") is None


def test_run_id_cannot_leave_the_data_folder():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain = Captain(root, spawn=lambda *a: 1, capture=lambda *a: {"code": 0, "stdout": ""})
        try:
            captain.peek("../victim")
        except PlanError as exc:
            assert "run id" in str(exc)
        else:
            raise AssertionError("path traversal was accepted")
        assert not (root.parent / "victim.json").exists()


def test_slash_peek_does_not_call_a_model():
    import tempfile
    spec = importlib.util.spec_from_file_location(
        "hermes_captain_entry",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    entry = importlib.util.module_from_spec(spec)
    entry.__package__ = "hermes_captain"
    sys.modules["hermes_captain_entry"] = entry
    spec.loader.exec_module(entry)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "run.log"
        log.write_text("compiling\n", encoding="utf-8")
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "agent": "claude", "status": "running", "pid": 1, "log": str(log),
        }), encoding="utf-8")
        entry._CAPTAIN = Captain(root, spawn=lambda *a: 1, capture=lambda *a: {"code": 0, "stdout": ""})
        first = entry.handle_command("peek abc")
        second = entry.handle_command("peek abc")
    assert "compiling" not in first
    assert second == "No change."


def test_question_is_not_rewritten_as_no_change():
    from hermes_captain.peek import hold_reply
    reply = "We are waiting on your API key. Which account should I use?"
    assert hold_reply({"speak": False}, reply) is None


def test_t3_pending_approval_shows_on_a_launched_run():
    import tempfile
    inspect = json.dumps({"data": {"thread": {"id": "th_1", "pendingRequests": [{"id": "req"}]}}})

    def capture(argv, cwd, timeout):
        if "inspect" in argv:
            return {"code": 0, "stdout": inspect, "error": ""}
        return {"code": 0, "stdout": "", "error": ""}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "abc.json").write_text(json.dumps({
            "id": "abc",
            "agent": "t3",
            "surface": "t3",
            "status": "launched",
            "handle": "th_1",
        }), encoding="utf-8")
        card = Captain(root, spawn=lambda *a: 1, capture=capture).peek("abc")
    assert card["stuck_on_approval"] is True
    assert card["note"] == "Stuck on approval."


def test_remote_peek_does_not_diff_this_machine():
    import tempfile
    seen = {}

    def capture(argv, cwd, timeout):
        seen["argv"] = argv
        return {"code": 0, "stdout": "?? REVIEW.md\n", "error": ""}

    def local_diff(cwd):
        raise AssertionError("diffed the local machine")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "run.log"
        log.write_text("ok\n", encoding="utf-8")
        (root / "abc.json").write_text(json.dumps({
            "id": "abc",
            "agent": "claude",
            "status": "running",
            "pid": 1,
            "log": str(log),
            "cwd": "/Users/dev/src/hermes-captain",
            "host": "dev@203.0.113.7",
        }), encoding="utf-8")
        card = Captain(root, spawn=lambda *a: 1, capture=capture, diff=local_diff).peek("abc")
    assert card["diff"] == "?? REVIEW.md"
    assert seen["argv"][0] == "ssh"
    assert "dev@203.0.113.7" in seen["argv"]


def test_stderr_does_not_break_the_thread_id():
    from hermes_captain.runner import _capture, _guess_handle
    result = _capture(
        ["python3", "-c", "import sys; print('{\\\"ok\\\": true, \\\"data\\\": {\\\"thread\\\": {\\\"id\\\": \\\"th_9\\\", \\\"title\\\": \\\"review\\\"}}}'); print('warning', file=sys.stderr)"],
        None,
        10,
    )
    assert "warning" not in result["stdout"]
    assert _guess_handle(result["stdout"]) == "th_9"


def test_wait_pings_in_seconds_not_fifteen_minutes():
    import tempfile
    import time
    from hermes_captain.runner import _spawn
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain = Captain(root)
        log = root / "sleep.log"
        pid = _spawn(["sleep", "2"], None, log)
        proc = getattr(_spawn, "_last", None)
        assert proc is not None
        captain._procs[pid] = proc
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "pid": pid, "surface": "cli", "status": "running",
        }), encoding="utf-8")
        armed = captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
    assert armed["model"] is False
    assert done["model"] is False
    assert done["ping"] == "done"
    assert done["seconds"] is not None and done["seconds"] < 6
    assert done["seconds"] < 15 * 60
    del time


def test_remote_cli_stop_closes_the_tmux_window_and_ends_our_wait():
    import tempfile
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "agent": "claude", "pid": 1, "host": "dev@203.0.113.7", "status": "running",
        }), encoding="utf-8")
        captain = Captain(root, spawn=lambda *a: 1, capture=lambda argv, *a: calls.append(argv) or {"code": 0, "stdout": ""})
        result = captain.stop("abc")
        saved = json.loads((root / "abc.json").read_text())
    assert result["ok"] is True and saved["status"] == "stopped"
    assert calls[0][0] == "ssh" and "tmux kill-window" in calls[0][-1] and "abc.exit" in calls[0][-1]


def test_named_cli_agents_do_not_skip_approval():
    hermes = plan_run("hermes", "fix the test")
    assert hermes.argv == ["hermes", "chat", "-q", "fix the test", "-Q"]
    assert "--yolo" not in hermes.argv
    claude = plan_run("claude-code", "fix the test", continue_session=True)
    assert claude.argv[:3] == ["claude", "--continue", "-p"]
    codex = plan_run("codecs", "fix the test")
    assert codex.argv[:2] == ["codex", "exec"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in codex.argv
    grok = plan_run("grock", "fix the test")
    assert grok.argv == ["grok", "--no-auto-update", "-p", "fix the test"]
    assert "--always-approve" not in grok.argv
    opened = plan_run("open-code", "fix the test")
    assert opened.argv == ["opencode", "run", "fix the test"]
    assert "-p" not in opened.argv
    py = plan_run("pycode", "fix the test")
    assert py.argv == ["pycode", "-p", "fix the test"]
    assert "dangerous" not in " ".join(py.argv)
    try:
        plan_run("hermes", "fix the test", continue_session=True)
    except PlanError:
        pass
    else:
        raise AssertionError("hermes continue was accepted")
    picked = plan_run("claude", "fix the test", model="sonnet")
    assert picked.argv[:4] == ["claude", "--model", "sonnet", "-p"]
    cheap = plan_run("hermes", "fix the test", model="haiku")
    assert cheap.argv[1:4] == ["chat", "-m", "haiku"]
    assert "--yolo" not in cheap.argv
    try:
        plan_run("codex", "fix the test", model="gpt")
    except PlanError:
        return
    raise AssertionError("codex model flag was accepted")


def test_finished_keep_pings_once():
    import tempfile
    from hermes_captain.runner import _spawn
    notes = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain = Captain(root)
        captain.notify = lambda run_id, record: notes.append((run_id, record.get("keep")))
        log = root / "done.log"
        pid = _spawn(["python3", "-c", ""], None, log)
        proc = getattr(_spawn, "_last", None)
        captain._procs[pid] = proc
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "pid": pid, "surface": "cli", "status": "running", "keep": True,
        }), encoding="utf-8")
        captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
    assert done["ping"] == "done"
    assert notes == [("abc", True)]


def test_finish_pings_without_being_asked():
    import tempfile
    from hermes_captain.runner import _spawn
    notes = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain = Captain(root)
        captain.notify = lambda run_id, record: notes.append(record.get("ping"))
        log = root / "done.log"
        pid = _spawn(["python3", "-c", ""], None, log)
        captain._procs[pid] = getattr(_spawn, "_last", None)
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "pid": pid, "surface": "cli", "status": "running", "keep": False, "title": "beta",
        }), encoding="utf-8")
        captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
    assert done["ping"] == "done"
    assert notes == ["done"]


def test_real_t3_thread_ids_are_accepted():
    handle = "thread:delegated-task:command%3Amcp%3A58968731-a581-4993-8fff-3cb51a5a132e%3Adelegate-task%3A2490cc31"
    plan = plan_wait(handle)
    assert plan.argv[plan.argv.index("--thread") + 1] == handle


def test_t3_timeout_on_stderr_keeps_waiting():
    import tempfile
    calls = {"n": 0}

    def capture(argv, cwd, timeout):
        if "wait" not in argv:
            return {"code": 0, "stdout": "", "stderr": ""}
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "code": 6,
                "stdout": "",
                "stderr": json.dumps({"ok": False, "error": {"code": "THREAD_WAIT_TIMEOUT"}}),
            }
        return {
            "code": 0,
            "stdout": json.dumps({"data": {"wait": {"outcome": "completed"}}}),
            "stderr": "",
        }

    notes = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain = Captain(root, capture=capture)
        captain.notify = lambda run_id, record: notes.append(record.get("ping"))
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "surface": "t3", "handle": "mcp:79982033-5d61-4be6-9794-8f203010edca",
            "status": "running", "keep": False, "title": "Lane 4",
        }), encoding="utf-8")
        captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
    assert calls["n"] == 2
    assert done["ping"] == "done"
    assert notes == ["done"]


HANDLE = "thread:delegated-task:command%3Amcp%3A58968731-a581-4993-8fff-3cb51a5a132e%3Adelegate-task%3A2490cc31"
MAC = "dev@203.0.113.7"


ANSWER = "Fixed the bug. 3 tests pass."


def _read_envelope(answer=ANSWER):
    """Shape of `t3code --json threads read --detail answers --last-turn`, checked against a real T3 0.1.7 run."""
    return json.dumps({"ok": True, "data": {
        "runtime": {"origin": "http://127.0.0.1:3773", "serverVersion": "0.1.7", "capabilities": {"threadStop": True}},
        "auth": {"source": "desktop"},
        "snapshotSequence": 60080,
        "project": {"id": "p1", "title": "caas", "workspaceRoot": "/Users/dev/Projects/caas"},
        "thread": {
            "id": "mcp:58968731-a581-4993-8fff-3cb51a5a132e", "title": "Lane 4", "runStatus": "idle",
            "turns": [{"index": 1, "turnId": "run:1", "state": "completed", "finalMessageId": "message:provider:2"}],
            "messages": [
                {"id": "message:user:1", "role": "user", "text": "Please fix the bug", "turnId": "run:1", "streaming": False},
                {"id": "message:provider:1", "role": "assistant", "text": "Looking at it now", "turnId": "run:1", "streaming": False},
                {"id": "message:provider:2", "role": "assistant", "text": answer, "turnId": "run:1", "streaming": False, "textTruncated": False},
            ],
        },
    }})


def _t3_captain(root, wait_replies, pending=False, read_out=None, thread=None):
    read_out = read_out or _read_envelope()
    calls = []
    waits = iter(wait_replies)

    def capture(argv, cwd, timeout):
        calls.append(argv)
        cmd = argv[-1]
        if " wait " in cmd:
            return next(waits)
        if "inspect" in cmd:
            req = [{"id": "r1"}] if pending else []
            return {"code": 0, "stdout": json.dumps({"data": {"thread": {"pendingRequests": req, **(thread or {})}}}), "stderr": ""}
        if "send" in cmd:
            return {"code": 0, "stdout": "{}", "stderr": ""}
        return {"code": 0, "stdout": read_out, "stderr": ""}

    captain = Captain(root, capture=capture)
    captain.retry_delay = 0
    (root / "abc.json").write_text(json.dumps({
        "id": "abc", "agent": "t3", "surface": "t3", "handle": HANDLE, "host": MAC,
        "status": "running", "session_key": "sess-1", "title": "Lane 4",
    }), encoding="utf-8")
    return captain, calls


TIMEOUT = {"code": 6, "stdout": "", "stderr": json.dumps({"ok": False, "error": {"code": "THREAD_WAIT_TIMEOUT"}})}
COMPLETED = {"code": 0, "stdout": json.dumps({"data": {"wait": {"outcome": "completed"}}}), "stderr": ""}


def test_vps_to_mac_finish_sends_the_result_once_after_a_timeout():
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, calls = _t3_captain(Path(tmp), [TIMEOUT, COMPLETED])
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
        saved = json.loads((Path(tmp) / "abc.json").read_text())
    assert len(sent) == 1  # the timeout did not report finished
    assert ANSWER in sent[0]
    for wrapper in ('"ok"', '"data"', "serverVersion", "capabilities", "Please fix the bug", "Looking at it now", "message:provider"):
        assert wrapper not in sent[0]
    assert "conductor_steer" in sent[0] and "Lane 4" in sent[0]
    assert saved["follow_up"] == "delivered"
    waits = [a for a in calls if "wait" in a[-1]]
    assert len(waits) == 2
    for argv in waits:
        assert argv[0] == "ssh" and "BatchMode=yes" in argv and MAC in argv
        assert '.local/bin' in argv[-1] and HANDLE in argv[-1].replace("'", "")
        assert not any(b in _flags(argv) for b in BANNED)


def test_answer_text_reads_the_final_assistant_message_only():
    from hermes_captain.runner import _answer_text
    assert _answer_text(_read_envelope("All done.")) == "All done."
    still_streaming = json.loads(_read_envelope())
    still_streaming["data"]["thread"]["messages"][-1]["streaming"] = True
    assert _answer_text(json.dumps(still_streaming)) == "Looking at it now"
    no_answer = json.loads(_read_envelope())
    no_answer["data"]["thread"]["messages"] = no_answer["data"]["thread"]["messages"][:1]
    assert _answer_text(json.dumps(no_answer)) == ""
    assert _answer_text("plain text answer\n") == "plain text answer"


def test_approval_pending_tells_the_user_and_blocks_steering():
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, calls = _t3_captain(Path(tmp), [COMPLETED], pending=True)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
        steered = captain.steer("abc", "keep going")
    assert "approval" in sent[0] and "Do not steer" in sent[0]
    assert "conductor_steer on run" not in sent[0]
    assert steered["ok"] is False
    assert not any("send" in a[-1] for a in calls)


def test_steer_goes_through_when_nothing_is_pending():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        captain, calls = _t3_captain(Path(tmp), [])
        steered = captain.steer("abc", "now add a test")
    assert steered["ok"] is True
    assert any("--if-busy" in a[-1] for a in calls)


def test_lost_wait_still_tells_the_user():
    import tempfile
    sent = []
    down = {"code": 255, "stdout": "", "stderr": "ssh: connect timed out"}
    with tempfile.TemporaryDirectory() as tmp:
        captain, calls = _t3_captain(Path(tmp), [down, down, down])
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
    assert done["ping"] == "wait failed"
    assert len(sent) == 1 and "lost track" in sent[0] and "may still be running" in sent[0]


def test_undelivered_message_is_resent_after_restart():
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [COMPLETED])
        captain.notify = lambda run_id, record: False
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
        assert json.loads((Path(tmp) / "abc.json").read_text())["follow_up"] == "undelivered"
        again = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": ""})
        again.notify = lambda run_id, record: sent.append(record["message"]) or True
        again.resume()
        assert json.loads((Path(tmp) / "abc.json").read_text())["follow_up"] == "delivered"
    assert len(sent) == 1 and "Fixed the bug" in sent[0]


def test_plugin_injects_the_result_into_the_asking_session():
    spec = importlib.util.spec_from_file_location("hermes_captain.entry", ROOT / "__init__.py")
    entry = importlib.util.module_from_spec(spec)
    sys.modules["hermes_captain.entry"] = entry
    spec.loader.exec_module(entry)
    got = {}

    class Ctx:
        def inject_message(self, text, session_key=None):
            got.update(text=text, key=session_key)
            return True

    entry._CTX = Ctx()
    record = {"message": "Conductor: Lane 4 finished. Fixed the bug.", "session_key": "sess-1"}
    assert entry._follow_up("abc", record) is True
    assert got == {"text": record["message"], "key": "sess-1"}
    entry._CTX = None
    assert entry._follow_up("abc", record) is False



def test_long_answer_keeps_the_end_and_says_what_was_cut():
    import tempfile
    sent = []
    answer = "HEAD" + "x" * 2500 + "THE CONCLUSION"
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [COMPLETED], read_out=_read_envelope(answer))
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
    cut = len(answer) - 1500
    assert f"[first {cut} characters cut]\n" in sent[0]
    assert sent[0].index("characters cut") < sent[0].index("xxx")
    assert "THE CONCLUSION" in sent[0] and "HEAD" not in sent[0]
    assert '"data"' not in sent[0] and "serverVersion" not in sent[0]


def test_short_answer_is_not_marked_as_cut():
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [COMPLETED])
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
    assert "characters cut" not in sent[0]


def _cli_record(root, pid, started):
    (root / "abc.json").write_text(json.dumps({
        "id": "abc", "pid": pid, "pid_started": started, "surface": "cli", "agent": "claude",
        "status": "running", "session_key": "sess-1", "title": "Lane 4",
    }), encoding="utf-8")


def test_stop_refuses_a_stale_or_unverifiable_cli_process():
    import tempfile
    from unittest.mock import patch

    for saved, current, status in (
        ("old", "new", "running"), ("", "current", "running"),
        ("saved", "", "running"), ("same", "same", "exited"),
        ("same", "same", "stopped"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _cli_record(root, 12345, saved)
            record = json.loads((root / "abc.json").read_text())
            record["status"] = status
            (root / "abc.json").write_text(json.dumps(record))
            captain = Captain(root)
            with patch.object(_runner, "_proc_start", return_value=current), \
                    patch.object(_runner.os, "killpg") as killpg, patch.object(_runner.os, "kill") as kill:
                assert captain.stop("abc")["ok"] is False
                killpg.assert_not_called()
                kill.assert_not_called()
            assert json.loads((root / "abc.json").read_text()) == record


def test_stop_refuses_a_changed_or_unavailable_process_group():
    import tempfile
    from unittest.mock import patch

    for group, starts, failure in (
        (99999, ["saved"], None), (12345, ["saved", "new"], None),
        (12345, ["saved", "saved"], ProcessLookupError()),
        (12345, ["saved", "saved"], PermissionError()),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _cli_record(root, 12345, "saved")
            before = (root / "abc.json").read_text()
            captain = Captain(root)
            with patch.object(_runner, "_proc_start", side_effect=starts), \
                    patch.object(_runner.os, "getpgid", return_value=group), \
                    patch.object(_runner.os, "killpg", side_effect=failure) as killpg, \
                    patch.object(_runner.os, "kill") as kill:
                assert captain.stop("abc")["ok"] is False
                if failure is None:
                    killpg.assert_not_called()
                kill.assert_not_called()  # never fall back to a possibly recycled PID
            assert (root / "abc.json").read_text() == before


def test_stop_signals_only_the_verified_live_cli_process_group():
    import tempfile
    import signal
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _cli_record(root, 12345, "saved")
        captain = Captain(root)
        with patch.object(_runner, "_proc_start", return_value="saved") as identity, \
                patch.object(_runner.os, "getpgid", return_value=12345), \
                patch.object(_runner.os, "killpg") as killpg, patch.object(_runner.os, "kill") as kill:
            assert captain.stop("abc")["ok"] is True
            assert identity.call_count == 2
            killpg.assert_called_once_with(12345, signal.SIGTERM)
            kill.assert_not_called()
        assert json.loads((root / "abc.json").read_text())["status"] == "stopped"


def test_restart_rearms_a_cli_wait_only_for_the_same_process():
    import tempfile
    from hermes_captain.runner import _proc_start, _spawn
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pid = _spawn(["sleep", "1"], None, root / "run.log")
        proc = getattr(_spawn, "_last", None)
        _cli_record(root, pid, _proc_start(pid))
        import threading
        threading.Thread(target=proc.wait, daemon=True).start()  # in real life init reaps it; here the test is the parent
        fresh = Captain(root)
        fresh.notify = lambda run_id, record: sent.append(record["message"]) or True
        fresh.resume()
        done = fresh.wait_done("abc", timeout=8)
    assert done["ping"] == "done"
    assert "finished" in sent[0]


def test_restart_does_not_attach_to_a_stranger_with_the_same_pid():
    import tempfile
    from hermes_captain.runner import _spawn
    for started in ("Mon Jan  1 00:00:00 2001", ""):  # a different start time, and a record from before start times were kept
        sent = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid = _spawn(["sleep", "30"], None, root / "run.log")
            proc = getattr(_spawn, "_last", None)
            _cli_record(root, pid, started)
            fresh = Captain(root)
            fresh.notify = lambda run_id, record: sent.append(record["message"]) or True
            fresh.resume()
            saved = json.loads((root / "abc.json").read_text())
            proc.terminate()
            proc.wait()
        assert "abc" not in fresh._waits
        assert saved["ping"] == "wait lost" and saved["follow_up"] == "delivered"
        assert len(sent) == 1 and "lost track" in sent[0] and "may still be running" in sent[0]


def test_supervise_picks_up_a_run_the_turn_can_no_longer_wait_on():
    import tempfile
    import time
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "surface": "t3", "agent": "t3", "handle": "mcp:1",
            "title": "proof", "status": "running", "session_key": "sess-1",
        }), encoding="utf-8")
        captain = Captain(root, capture=lambda argv, cwd, timeout: {
            "code": 0, "stdout": _read_envelope("pong from the mac"),
        })
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        assert captain.supervise(interval=0.05) is True
        assert Captain(root).supervise(interval=0.05) is False
        deadline = time.time() + 3
        while not sent and time.time() < deadline:
            time.sleep(0.05)
        captain._supervise_stop.set()
    assert sent and "pong from the mac" in sent[0]
    assert '"data"' not in sent[0]


def _ago(minutes):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _cli_wait(root, argv, age_minutes=None, notes=None):
    """Run a real child with a health tick every 0.1s. The log is backdated so the run looks silent."""
    import os
    import time
    from hermes_captain.runner import _spawn
    captain = Captain(root)
    captain.health_every = 0.1
    captain.notify = lambda run_id, record: (notes if notes is not None else []).append(record["message"]) or True
    log = root / "run.log"
    pid = _spawn(argv, None, log)
    captain._procs[pid] = getattr(_spawn, "_last", None)
    if age_minutes:
        old = time.time() - age_minutes * 60
        os.utime(log, (old, old))
    (root / "abc.json").write_text(json.dumps({
        "id": "abc", "pid": pid, "surface": "cli", "agent": "claude", "status": "running",
        "title": "beta", "log": str(log),
    }), encoding="utf-8")
    captain.arm_wait("abc")
    return captain.wait_done("abc", timeout=10)


def test_health_words():
    from hermes_captain.peek import run_health
    assert run_health(alive=True, idle=5)[0] == "working"
    assert run_health(alive=True, idle=None)[0] == "working"
    assert run_health(alive=True, idle=179)[0] == "working"
    assert run_health(alive=True, idle=180)[0] == "quiet"
    assert run_health(alive=True, idle=599)[0] == "quiet"
    assert run_health(alive=True, idle=600) == ("stalled", "no new output for 10 minutes")
    assert run_health(alive=True, idle=9999, stuck=True)[0] == "needs-approval"
    assert run_health(alive=False, idle=9999)[0] == "done"
    assert run_health(alive=False, idle=0, reason="exited with code 2") == ("errored", "exited with code 2")


def test_error_reason_names_which_failure():
    from hermes_captain.peek import error_reason
    assert error_reason(1, "boom") == "exited with code 1"
    assert error_reason(0, "") == ""
    assert error_reason(None, "Claude AI usage limit reached|1700000000") == "usage limit reached"
    assert error_reason(1, "Invalid API key · Please run /login") == "login failed"


def test_peek_reports_health_from_the_log_mtime():
    import os
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "run.log"
        log.write_text("compiling\n", encoding="utf-8")
        (root / "abc.json").write_text(json.dumps({
            "id": "abc", "agent": "claude", "status": "running", "pid": 1, "log": str(log),
        }), encoding="utf-8")
        captain = Captain(root, spawn=lambda *a: 1, capture=lambda *a: {"code": 0, "stdout": ""})
        first = captain.peek("abc")
        for minutes, word in ((4, "quiet"), (11, "stalled")):
            old = time.time() - minutes * 60
            os.utime(log, (old, old))
            card = captain.peek("abc")
            assert card["health"] == word
            assert card["speak"] is True  # a stall is news even though the log did not change
        again = captain.peek("abc")
    assert first["health"] == "working"
    assert card["note"] == "no new output for 11 minutes"
    assert again["speak"] is False and again["health"] == "stalled"


def test_peek_health_for_finished_and_approval_runs():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, text, extra, word, why in (
            ("a", "all good\n", {}, "done", "finished"),
            ("b", "oops\n", {"exit_code": 2}, "errored", "exited with code 2"),
            ("c", "You have hit your limit\n", {}, "errored", "usage limit reached"),
            ("d", "Not logged in · Please run /login\n", {}, "errored", "login failed"),
        ):
            log = root / f"{name}.log"
            log.write_text(text, encoding="utf-8")
            (root / f"{name}.json").write_text(json.dumps({
                "id": name, "agent": "claude", "status": "exited", "log": str(log), **extra,
            }), encoding="utf-8")
            card = Captain(root, capture=lambda *a: {"code": 0, "stdout": ""}).peek(name)
            assert (card["health"], card["note"]) == (word, why if word == "errored" else card["note"]), name
        inspect = json.dumps({"data": {"thread": {"pendingRequests": [{"id": "r"}], "runStatus": "idle"}}})
        (root / "t.json").write_text(json.dumps({
            "id": "t", "agent": "t3", "surface": "t3", "status": "launched", "handle": "th_1",
        }), encoding="utf-8")
        card = Captain(root, capture=lambda argv, c, t: {"code": 0, "stdout": inspect if "inspect" in argv else ""}).peek("t")
    assert card["health"] == "needs-approval"


def test_stall_pings_once_and_quiet_does_not_ping():
    import tempfile
    stalled, quiet = [], []
    with tempfile.TemporaryDirectory() as tmp:
        done = _cli_wait(Path(tmp), ["sleep", "1"], age_minutes=11, notes=stalled)
    with tempfile.TemporaryDirectory() as tmp:
        _cli_wait(Path(tmp), ["sleep", "1"], age_minutes=4, notes=quiet)
    assert done["ping"] == "done"
    assert len(stalled) == 2  # one stall ping, one finish ping
    assert "stalled" in stalled[0] and "no new output for 11 minutes" in stalled[0]
    assert "finished" in stalled[1]
    assert len(quiet) == 1 and "finished" in quiet[0]


def test_cli_exit_code_and_login_failure_ping_as_errored():
    import tempfile
    for argv, expect in (
        (["python3", "-c", "import sys; sys.exit(3)"], "exited with code 3"),
        (["python3", "-c", "print('Invalid API key · Please run /login')"], "login failed"),
        (["python3", "-c", "print('Claude AI usage limit reached|1700000000')"], "usage limit reached"),
    ):
        notes = []
        with tempfile.TemporaryDirectory() as tmp:
            _cli_wait(Path(tmp), argv, notes=notes)
        assert len(notes) == 1 and "errored" in notes[0] and expect in notes[0], (argv, notes)
    ok = []
    with tempfile.TemporaryDirectory() as tmp:
        _cli_wait(Path(tmp), ["python3", "-c", "print('fine')"], notes=ok)
    assert "finished" in ok[0] and "errored" not in ok[0]


def test_t3_stall_pings_once_then_the_finish():
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, calls = _t3_captain(
            Path(tmp), [TIMEOUT, TIMEOUT, COMPLETED], thread={"runStatus": "running", "updatedAt": _ago(15)},
        )
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
    assert len(sent) == 2
    assert "stalled" in sent[0] and "15 minutes" in sent[0]
    assert "finished" in sent[1]


def test_t3_busy_thread_does_not_ping_and_failed_run_is_errored():
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(
            Path(tmp), [TIMEOUT, COMPLETED], thread={"runStatus": "running", "updatedAt": _ago(1)},
        )
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
    assert len(sent) == 1 and "finished" in sent[0]
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, calls = _t3_captain(Path(tmp), [TIMEOUT, COMPLETED], thread={"runStatus": "failed"})
        captain.notify = lambda run_id, record: failed.append(record["message"]) or True
        captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
    assert done["ping"] == "errored"
    assert len(failed) == 1 and "errored: the thread run failed" in failed[0]
    assert len([a for a in calls if " wait " in a[-1]]) == 1


def test_build_prompt_gives_the_structure():
    from hermes_captain.plan import build_prompt
    prompt = build_prompt("make login redirect", ["app/login.py", "tests/test_login.py"], "pytest tests/test_login.py passes")
    for part in ("Goal: make login redirect", "- app/login.py", "- tests/test_login.py", "Done when: pytest tests/test_login.py passes",
                 "smallest change that works", "No extra features", "No new dependencies", "final answer under 100 words: what changed, the verify output, what is left"):
        assert part in prompt, part
    bare = build_prompt("fix it")
    assert "Files in scope" not in bare and "Done when" not in bare and "smallest change" in bare
    assert "- a.py" in build_prompt("x", "a.py")
    try:
        build_prompt("  ", ["a.py"])
    except PlanError as exc:
        assert str(exc) == "goal is missing"
    else:
        raise AssertionError("empty goal was accepted")


def _entry():
    spec = importlib.util.spec_from_file_location(
        "hermes_captain_entry", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
    )
    entry = importlib.util.module_from_spec(spec)
    entry.__package__ = "hermes_captain"
    sys.modules["hermes_captain_entry"] = entry
    spec.loader.exec_module(entry)
    return entry


def test_conductor_run_builds_the_kickoff_and_keeps_the_free_form_path():
    import tempfile
    seen = []

    def capture(argv, cwd, timeout):
        seen.append(argv[argv.index("--prompt") + 1])
        return {"code": 0, "stdout": "{}", "error": ""}

    entry = _entry()
    with tempfile.TemporaryDirectory() as tmp:
        entry._CAPTAIN = Captain(Path(tmp), capture=capture)
        built = json.loads(entry._run({"agent": "t3", "goal": "fix login", "files": ["a.py"], "done_when": "tests pass"}))
        free = json.loads(entry._run({"agent": "t3", "prompt": "just do the thing"}))
        none = json.loads(entry._run({"agent": "t3"}))
        no_goal = json.loads(entry._run({"agent": "t3", "files": ["a.py"]}))
    assert built["ok"] and free["ok"]
    assert "Goal: fix login" in seen[0] and "- a.py" in seen[0] and "Done when: tests pass" in seen[0]
    assert seen[1] == "just do the thing"
    assert none == {"ok": False, "error": "goal is missing"} == no_goal
    assert len(seen) == 2


def _steer_with(root, thread, last_active=None):
    captain, calls = _t3_captain(root, [], thread=thread)
    if last_active is not None:
        record = json.loads((root / "abc.json").read_text())
        record["last_active"] = last_active
        (root / "abc.json").write_text(json.dumps(record), encoding="utf-8")
    return captain.steer("abc", "next step"), json.loads((root / "abc.json").read_text())


def test_steer_reports_cache_and_advises_only_when_cold_and_long():
    import tempfile
    import time
    from hermes_captain.peek import LONG_THREAD
    with tempfile.TemporaryDirectory() as tmp:
        warm, saved = _steer_with(Path(tmp), {"updatedAt": _ago(1), "turnCount": 99})
    assert warm["ok"] is True and warm["cache_warm"] is True and 50 <= warm["idle_seconds"] <= 70
    assert "advice" not in warm
    assert abs(saved["last_active"] - time.time()) < 5
    with tempfile.TemporaryDirectory() as tmp:
        cold, _ = _steer_with(Path(tmp), {"updatedAt": _ago(10), "turnCount": LONG_THREAD})
    assert cold["ok"] is True and cold["cache_warm"] is False and cold["idle_seconds"] >= 590
    assert "re-reads the whole history" in cold["advice"] and "two-line summary" in cold["advice"]
    assert "saving" not in cold["advice"] and "%" not in cold["advice"]
    with tempfile.TemporaryDirectory() as tmp:
        short, _ = _steer_with(Path(tmp), {"updatedAt": _ago(10), "turnCount": LONG_THREAD - 1})
    assert short["cache_warm"] is False and "advice" not in short
    with tempfile.TemporaryDirectory() as tmp:
        by_messages, _ = _steer_with(Path(tmp), {"updatedAt": _ago(10), "messageCount": LONG_THREAD * 2})
    assert "advice" in by_messages


def test_steer_falls_back_to_last_active_and_never_guesses():
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as tmp:
        cold, _ = _steer_with(Path(tmp), {"turnCount": 99}, last_active=time.time() - 3600)
    assert cold["cache_warm"] is False and cold["idle_seconds"] >= 3590 and "advice" in cold
    with tempfile.TemporaryDirectory() as tmp:
        unknown, _ = _steer_with(Path(tmp), {"turnCount": 99})
    assert unknown["cache_warm"] is None and unknown["idle_seconds"] is None and "advice" not in unknown


def _t3_finish(thread, waits):
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), waits, thread=thread)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
        card = captain.peek("abc")
    return done, sent, card


def test_failed_thread_reports_its_last_error_cut_to_160():
    error = "Claude Code can't use your Claude subscription, usually because a payment is past due \u00b7 Pay or renew at https://example.com/" + "x" * 200
    done, sent, card = _t3_finish(
        {"runStatus": "failed", "lastError": error, "lastErrorClass": "provider_error", "lastErrorCode": "api_error_403"},
        [TIMEOUT, COMPLETED],
    )
    assert done["ping"] == "errored"
    assert len(sent) == 1 and f"errored: {error[:160]} (run" in sent[0] and error[:161] not in sent[0]
    assert card["health"] == "errored" and card["note"] == error[:160]
    done, sent, card = _t3_finish({"runStatus": "failed", "lastError": "  "}, [TIMEOUT, COMPLETED])
    assert "errored: the thread run failed" in sent[0] and card["note"] == "the thread run failed"


def test_interrupted_thread_is_errored_and_never_a_normal_finish():
    for waits in ([TIMEOUT, COMPLETED], [COMPLETED]):  # the tick sees it, or wait calls it completed
        done, sent, card = _t3_finish({"runStatus": "interrupted"}, waits)
        assert len(sent) == 1 and "errored: the run was interrupted" in sent[0], waits
        assert "finished" not in sent[0]
        assert card["health"] == "errored" and card["note"] == "the run was interrupted"


def _finish_text(saved=None, **extra):
    """The finish message for a done T3 run, with the saved mode and any per-run fields."""
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, _ = _t3_captain(root, [COMPLETED])
        record = json.loads((root / "abc.json").read_text())
        (root / "abc.json").write_text(json.dumps({**record, **extra}), encoding="utf-8")
        if saved:
            captain.set_mode(saved)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
    return sent[0]


def test_mode_defaults_to_advise_and_advise_sends_nothing():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert Captain(Path(tmp)).get_mode() == "advise"
    text = _finish_text()
    assert "recommend one next step in one line" in text
    assert "Do not call conductor_steer or conductor_run" in text
    assert "send one follow-up" not in text and "round" not in text


def test_drive_keeps_steering_and_counts_rounds():
    text = _finish_text("drive")
    assert "send one follow-up with conductor_steer on run abc (round 1 of 5)" in text
    assert "stalled, errored, needs approval, or lost" in text
    assert "round 3 of 5" in _finish_text("drive", rounds=2)
    assert "round 5 of 5" in _finish_text("drive", rounds=4)


def test_drive_stops_at_the_cap():
    from hermes_captain.runner import MAX_ROUNDS
    assert MAX_ROUNDS == 5
    text = _finish_text("drive", rounds=MAX_ROUNDS)
    assert "used all 5 follow-ups" in text and "Do not call conductor_steer or conductor_run" in text
    assert "send one follow-up" not in text


def test_per_run_mode_beats_the_saved_setting():
    assert "send one follow-up" not in _finish_text("drive", follow_mode="advise")
    assert "send one follow-up" in _finish_text("advise", follow_mode="drive")


def test_mode_is_saved_and_a_bad_value_is_refused():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert Captain(root).set_mode("Drive") == "drive"
        assert Captain(root).get_mode() == "drive"  # a fresh process reads the file
        for bad in ("yolo", "", "drive; advise"):
            try:
                Captain(root).set_mode(bad)
            except PlanError as exc:
                assert "drive or advise" in str(exc)
            else:
                raise AssertionError(f"{bad!r} was accepted")
        assert Captain(root).get_mode() == "drive"
        (root / "mode.json").write_text("not json", encoding="utf-8")
        assert Captain(root).get_mode() == "advise"
        try:
            Captain(root, spawn=lambda *a: (_ for _ in ()).throw(AssertionError("spawned"))).run("claude", "x", mode="yolo")
        except PlanError:
            pass
        else:
            raise AssertionError("run accepted a bad mode")
        saved = Captain(root, capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""}).run("t3", "x", mode="drive")
        plain = Captain(root, capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""}).run("t3", "x")
    assert saved["follow_mode"] == "drive" and "follow_mode" not in plain


def test_steer_counts_a_round_only_when_it_is_sent():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, _ = _t3_captain(root, [])
        captain.steer("abc", "one")
        captain.steer("abc", "two")
        assert json.loads((root / "abc.json").read_text())["rounds"] == 2
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, _ = _t3_captain(root, [], pending=True)
        captain.steer("abc", "blocked")
        assert "rounds" not in json.loads((root / "abc.json").read_text())


def test_mode_tool_and_slash_command():
    import tempfile
    entry = _entry()
    with tempfile.TemporaryDirectory() as tmp:
        entry._CAPTAIN = Captain(Path(tmp))
        assert entry.handle_command("mode") == "Mode: advise"
        assert entry.handle_command("mode drive") == "Mode: drive"
        assert entry.handle_command("mode") == "Mode: drive"
        assert "drive or advise" in entry.handle_command("mode yolo")
        assert entry.handle_command("mode") == "Mode: drive"
        assert json.loads(entry._mode({})) == {"ok": True, "mode": "drive"}
        assert entry.handle_command("mode advise") == "Mode: advise"
        # the model can read the mode, never set it: drive lets runs continue unattended
        assert json.loads(entry._mode({"mode": "drive"})) == {"ok": True, "mode": "advise"}
        assert entry.handle_command("mode") == "Mode: advise"
        entry._CAPTAIN = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""})
        ran = json.loads(entry._run({"agent": "t3", "prompt": "x", "mode": "drive"}))
    assert "follow_mode" not in ran["run"]
    assert "mode" not in entry._SCHEMAS["conductor_run"]["parameters"]["properties"]
    assert entry._SCHEMAS["conductor_mode"]["parameters"]["properties"] == {}


def _board(captain, fail=None, gate=None):
    """Route hermes calls to a list instead of a real board. fail: 'raise' or 'code'. gate: an Event the call waits on."""
    calls = []
    real = captain.capture

    def capture(argv, cwd, timeout):
        if argv[0] != "hermes":
            return real(argv, cwd, timeout)
        calls.append(argv)
        if gate is not None:
            gate.wait(10)
        if fail == "raise":
            raise OSError("no hermes")
        return {"code": 1 if fail == "code" else 0, "stdout": "", "error": "board down" if fail else ""}

    captain.capture = capture
    return calls


def _board_run(thread=None, waits=(COMPLETED,), card="t_42", pending=False, **board):
    """Wait out a T3 run. Returns (hermes argvs, messages sent, captain)."""
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, _ = _t3_captain(root, list(waits), pending=pending, thread=thread)
        if card:
            record = json.loads((root / "abc.json").read_text())
            (root / "abc.json").write_text(json.dumps({**record, "card": card}), encoding="utf-8")
        calls = _board(captain, **board)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
        for thread_ in captain._kanban_threads:
            thread_.join(5)
    return calls, sent, captain


def test_card_gets_start_comment_stall_comment_and_the_answer():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""})
        calls = _board(captain)
        captain.run("t3", "review", host=MAC, card="t_42")
        for t in captain._kanban_threads:
            t.join(5)
    assert calls == [["hermes", "kanban", "comment", "t_42", f"Conductor: started: t3 {MAC}"]]
    calls, sent, _ = _board_run(thread={"runStatus": "running", "updatedAt": _ago(15)}, waits=[TIMEOUT, TIMEOUT, COMPLETED])
    verbs = [c[2] for c in calls]
    assert verbs == ["comment", "comment"]
    assert calls[0][3:] == ["t_42", "Conductor: stalled: no new output for 15 minutes"]
    assert calls[1][3:] == ["t_42", f"Conductor: {ANSWER}"]


def test_card_is_blocked_on_errored_and_on_approval_and_never_completed():
    calls, _, _ = _board_run(thread={"runStatus": "failed", "lastError": "payment is past due"})
    assert calls == [["hermes", "kanban", "block", "t_42", "Conductor: payment is past due"]]
    every = list(calls)
    calls, _, _ = _board_run(pending=True)
    assert calls == [["hermes", "kanban", "block", "--kind", "needs_input", "t_42",
                      "Conductor: waiting on an approval; approve it on that machine"]]
    every += calls
    calls, _, _ = _board_run(thread={"runStatus": "running", "updatedAt": _ago(15)}, waits=[TIMEOUT, COMPLETED])
    every += calls
    done, _, _ = _board_run()
    every += done
    assert len(every) >= 4
    assert all(c[:2] == ["hermes", "kanban"] and c[2] in ("comment", "block") for c in every)
    assert not any("complete" in part for c in every for part in c[:3])


def test_no_card_means_no_kanban_calls():
    import tempfile
    for kwargs in (dict(thread={"runStatus": "failed"}), dict(pending=True), dict()):
        calls, sent, _ = _board_run(card=None, **kwargs)
        assert calls == [] and len(sent) == 1
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""})
        calls = _board(captain)
        captain.run("t3", "review")
        for t in captain._kanban_threads:
            t.join(2)
        for bad in ("--evil", "a b", "x" * 80):
            try:
                captain.run("t3", "review", card=bad)
            except PlanError:
                pass
            else:
                raise AssertionError(f"{bad!r} accepted as a card")
    assert calls == []


def test_a_failing_board_never_breaks_the_run_and_is_logged_once():
    import logging
    import threading
    seen = []

    class Grab(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    log = logging.getLogger("hermes_captain.runner")
    handler = Grab()
    log.addHandler(handler)
    try:
        for fail in ("raise", "code"):
            seen.clear()
            calls, sent, captain = _board_run(thread={"runStatus": "running", "updatedAt": _ago(15)}, waits=[TIMEOUT, TIMEOUT, COMPLETED], fail=fail)
            assert len(calls) == 2 and len(sent) == 2 and "finished" in sent[1]
            assert len([m for m in seen if "kanban" in m]) == 1, seen
    finally:
        log.removeHandler(handler)
    # a board that hangs must not hold up the ping
    gate = threading.Event()
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, _ = _t3_captain(root, [COMPLETED])
        record = json.loads((root / "abc.json").read_text())
        (root / "abc.json").write_text(json.dumps({**record, "card": "t_42"}), encoding="utf-8")
        _board(captain, gate=gate)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        done = captain.wait_done("abc", timeout=8)
        pinged_while_hung = bool(sent)
        gate.set()
    assert done["ping"] == "done" and pinged_while_hung


def test_reset_time_parses_three_formats_and_gives_up_on_the_rest():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from hermes_captain.peek import reset_time
    denver = ZoneInfo("America/Denver")
    now = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("UTC")).timestamp()  # 05:00 in Denver
    at = reset_time("You've hit your limit \u00b7 resets 3:00 PM (America/Denver)", now)
    assert at == datetime(2026, 1, 15, 15, 0, tzinfo=denver).timestamp()
    assert reset_time("limit reached, resets 12am (America/Denver)", now) == datetime(2026, 1, 16, 0, 0, tzinfo=denver).timestamp()
    late = datetime(2026, 1, 15, 16, 0, tzinfo=denver).timestamp()
    assert reset_time("resets 3pm (America/Denver)", late) == datetime(2026, 1, 16, 15, 0, tzinfo=denver).timestamp()
    local = reset_time("Usage limit reached. Your limit resets at 3pm", now)
    assert datetime.fromtimestamp(local).hour == 15 and 0 < local - now <= 86400
    assert reset_time("usage limit reached. Try again in 2 hours", now) == now + 7200
    assert reset_time("quota exceeded, try again in 30 minutes", now) == now + 1800
    for nothing in ("usage limit reached", "", "resets at noonish", "resets 13pm", "resets 3:75pm"):
        assert reset_time(nothing, now) is None, nothing


def _limit_run(lasterr=None, output=None):
    """A run that hit a usage limit. T3 when lasterr is given, else a CLI that printed output."""
    import tempfile
    if lasterr is not None:
        sent = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captain, _ = _t3_captain(root, [COMPLETED], thread={"runStatus": "failed", "lastError": lasterr})
            captain.notify = lambda run_id, record: sent.append(record["message"]) or True
            captain.arm_wait("abc")
            captain.wait_done("abc", timeout=8)
            return sent[0], json.loads((root / "abc.json").read_text())
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        _cli_wait(Path(tmp), ["python3", "-c", f"print({output!r})"], notes=sent)
        return sent[0], json.loads((Path(tmp) / "abc.json").read_text())


def test_usage_limit_stores_resume_at_or_says_the_time_is_unknown():
    import time
    text, saved = _limit_run(lasterr="You've hit your limit \u00b7 resets 3pm (America/Denver)")
    assert "errored" in text and "It should reset around" in text and "unknown" not in text
    assert saved["resume_at"] > time.time() and "resumed_at" not in saved
    text, saved = _limit_run(output="Claude AI usage limit reached. Try again in 2 hours")
    assert 7100 < saved["resume_at"] - time.time() <= 7200 and "It should reset around" in text
    text, saved = _limit_run(output="Claude AI usage limit reached")
    assert "Reset time unknown." in text and "resume_at" not in saved
    text, saved = _limit_run(output="boom")
    assert "unknown" not in text and "resume_at" not in saved


def _nudges(root, **extra):
    import time
    captain, _ = _t3_captain(root, [])
    record = json.loads((root / "abc.json").read_text())
    (root / "abc.json").write_text(json.dumps({
        **record, "pinged_at": time.time() - 60, "follow_up": "delivered", "message": "old", "ping": "errored", **extra,
    }), encoding="utf-8")
    sent = []
    captain.notify = lambda run_id, record: sent.append(record["message"]) or True
    return captain, sent


def test_resume_nudge_fires_once_and_not_before_the_reset():
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, sent = _nudges(root, resume_at=time.time() + 3600)
        captain.resume()
        assert sent == []  # not yet
        saved = json.loads((root / "abc.json").read_text())
        saved["resume_at"] = time.time() - 1
        (root / "abc.json").write_text(json.dumps(saved), encoding="utf-8")
        for _ in range(3):
            captain.resume()
        assert len(sent) == 1
        assert "Lane 4 hit a usage limit and it should have reset" in sent[0]
        assert "conductor_steer on run abc: 'Continue where you left off.'" in sent[0]
        assert json.loads((root / "abc.json").read_text())["resumed_at"] > 0
    with tempfile.TemporaryDirectory() as tmp:
        captain, sent = _nudges(Path(tmp))  # no resume_at at all
        captain.resume()
        assert sent == []


def test_resume_nudge_wording_follows_the_mode_and_the_cap():
    import tempfile
    import time
    due = {"resume_at": time.time() - 1}
    texts = {}
    for name, extra in (
        ("advise", {}),
        ("drive", {"follow_mode": "drive"}),
        ("cap", {"follow_mode": "drive", "rounds": 5}),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            captain, sent = _nudges(Path(tmp), **due, **extra)
            captain.resume()
            texts[name] = sent[0]
    assert "Ask the user first whether to resume it with conductor_steer" in texts["advise"]
    assert "Resume it with conductor_steer on run abc" in texts["drive"] and "Ask the user" not in texts["drive"]
    assert "used all 5 follow-ups" in texts["cap"] and "Do not call conductor_steer or conductor_run" in texts["cap"]
    assert "Resume it" not in texts["cap"]
    with tempfile.TemporaryDirectory() as tmp:  # a CLI run cannot be steered, so it is told to continue
        captain, sent = _nudges(Path(tmp), **due, follow_mode="drive", surface="cli", handle="", agent="claude", cwd="/work")
        captain.resume()
    assert "conductor_run (agent claude, continue true, cwd /work)" in sent[0] and "conductor_steer" not in sent[0]


def _push_wait(root, argv, events, before_exit=None, host=None, capture=None):
    """A real child with a push-wired record. `before_exit` runs on a thread while the child is still alive."""
    import threading
    from hermes_captain.runner import _spawn
    captain = Captain(root, capture=capture)
    captain.events_dir = root / "events"
    captain.events_dir.mkdir()
    captain.health_every = 100
    sent = []
    captain.notify = lambda run_id, record: sent.append(record["message"]) or True
    pid = _spawn(argv, None, root / "run.log")
    proc = getattr(_spawn, "_last", None)
    captain._procs[pid] = proc
    record = {"id": "abc", "pid": pid, "surface": "cli", "agent": "claude", "status": "running",
              "title": "beta", "log": str(root / "run.log"), "events": events}
    if host:
        record["host"] = host
    (root / "abc.json").write_text(json.dumps(record), encoding="utf-8")
    if before_exit:
        threading.Timer(0.5, before_exit, args=(captain.events_dir,)).start()
    captain.arm_wait("abc")
    done = captain.wait_done("abc", timeout=8)
    alive = proc.poll() is None
    if alive:
        proc.kill()
    proc.wait()
    return captain, done, sent, alive


def _event(kind, message=""):
    return json.dumps({"agent": "claude", "kind": kind, "time": 1, "message": message})


def test_an_event_file_wakes_the_wait_before_the_process_exits_and_is_deleted():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write = lambda d: (d / "abc.done.json").write_text(_event("done", "Shipped it."), encoding="utf-8")
        _, done, sent, alive = _push_wait(root, ["sleep", "30"], "push", before_exit=write)
        left = list((root / "events").iterdir())
    assert alive, "the wait should have ended while the agent was still running"
    assert done["ping"] == "done" and done["seconds"] < 5
    assert len(sent) == 1 and "finished" in sent[0] and "Shipped it." in sent[0]
    assert left == []
    with tempfile.TemporaryDirectory() as tmp:
        write = lambda d: (d / "abc.needs-input.json").write_text(_event("needs-input", "Claude needs your permission to use Bash"), encoding="utf-8")
        captain, done, sent, alive = _push_wait(Path(tmp), ["sleep", "30"], "push", before_exit=write)
        saved = json.loads((Path(tmp) / "abc.json").read_text())
    assert alive and done["ping"] == "needs-attention"
    assert "approval" in sent[0] and "needs your permission to use Bash" in sent[0] and "Do not steer" in sent[0]
    assert saved["event"] == "needs-input"


def test_broken_missing_or_unwired_hooks_fall_back_to_polling():
    import tempfile
    quick = ["python3", "-c", "import time; time.sleep(1.5)"]
    with tempfile.TemporaryDirectory() as tmp:  # push wired, the hook never fired
        _, done, sent, _ = _push_wait(Path(tmp), quick, "push")
    assert done["ping"] == "done" and len(sent) == 1 and "finished" in sent[0]
    with tempfile.TemporaryDirectory() as tmp:  # the hook wrote garbage: ignored, and not left behind
        root = Path(tmp)
        junk = lambda d: (d / "abc.done.json").write_text("{not json", encoding="utf-8")
        _, done, sent, alive = _push_wait(root, quick, "push", before_exit=junk)
        left = list((root / "events").iterdir())
    assert not alive and done["ping"] == "done" and done["seconds"] >= 1.0  # the exit ended it, not the junk
    assert len(sent) == 1 and left == []
    with tempfile.TemporaryDirectory() as tmp:  # an unwired agent never looks at the folder
        real = lambda d: (d / "abc.done.json").write_text(_event("done"), encoding="utf-8")
        _, done, _, alive = _push_wait(Path(tmp), quick, "polling", before_exit=real)
    assert not alive and done["seconds"] >= 1.0


def test_remote_event_wait_is_one_ssh_command_that_wakes_the_wait():
    import shlex
    import subprocess
    import tempfile
    import time
    from hermes_captain.plan import plan_event_wait
    argv = plan_event_wait("abc", MAC, 60).argv
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"] and MAC in argv
    remote = shlex.split(argv[-1])
    script = remote[-1]
    for part in ("test -e", 'cat "$f"', "rm -f", ".conductor/events", "abc.$k.json", "-ge 60", "sleep 1"):
        assert part in script, part
    try:
        plan_event_wait("../x", MAC, 60)
    except PlanError:
        pass
    else:
        raise AssertionError("bad run id accepted")
    with tempfile.TemporaryDirectory() as tmp:  # run the very script ssh would run
        folder = Path(tmp)
        env = {"PATH": "/usr/bin:/bin", "CONDUCTOR_EVENTS": str(folder)}
        (folder / "abc.done.json").write_text(_event("done", "ok"), encoding="utf-8")
        (folder / "abc.needs-input.json").write_text(_event("needs-input"), encoding="utf-8")
        got = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        assert got.returncode == 0 and json.loads(got.stdout)["kind"] == "done"
        assert list(folder.iterdir()) == []
        start = time.time()
        short = shlex.split(plan_event_wait("abc", MAC, 1).argv[-1])[-1]
        none = subprocess.run(["bash", "-c", short], env=env, capture_output=True, text=True)
        assert none.returncode == 6 and none.stdout == "" and 1 <= time.time() - start < 4
    seen = []

    def capture(argv, cwd, timeout):
        seen.append((argv, timeout))
        return {"code": 0, "stdout": _event("done", "from the mac"), "stderr": ""}

    with tempfile.TemporaryDirectory() as tmp:
        _, done, sent, alive = _push_wait(Path(tmp), ["sleep", "30"], "push", host=MAC, capture=capture)
    assert alive and done["ping"] == "done" and "from the mac" in sent[0]
    assert seen[0][0][0] == "ssh" and MAC in seen[0][0] and "test -e" in seen[0][0][-1] and seen[0][1] == 75


def _run_hook(command, env, stdin="", extra=()):
    import subprocess
    return subprocess.run(command, shell=isinstance(command, str), input=stdin, env=env, capture_output=True, text=True, timeout=20)


def test_claude_gets_per_run_settings_with_only_our_two_hooks():
    import os
    import tempfile
    assert plan_run("claude", "x").argv == ["claude", "-p", "x"]  # no run, no wiring
    plan = plan_run("claude", "x", run_id="abc", model="sonnet")
    assert plan.push == "push"
    assert plan.argv[:4] == ["env", "CONDUCTOR_RUN=abc", "claude", "--settings"]
    assert plan.argv[5:] == ["--model", "sonnet", "-p", "x"]
    settings = json.loads(plan.argv[4])
    assert list(settings) == ["hooks"] and set(settings["hooks"]) == {"Stop", "PermissionRequest", "Notification"}
    commands = {h["command"] for groups in settings["hooks"].values() for g in groups for h in g["hooks"]}
    assert len(commands) == 2
    assert {c.rsplit(" ", 1)[1] for c in commands} == {"done", "needs-input"}
    assert {c.split(" -c ")[0] for c in commands} == {"python3"}
    everything = " ".join(plan.argv)
    for theirs in (".claude", "settings.json", ".codex", ".cursor", "hooks.json", "setting-sources", "/Users/"):
        assert theirs not in everything, theirs
    stop = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
    with tempfile.TemporaryDirectory() as tmp:  # the quoted command really runs, and reads stdin
        env = {**os.environ, "CONDUCTOR_RUN": "abc", "CONDUCTOR_EVENTS": tmp}
        _run_hook(stop, env, stdin=json.dumps({"hook_event_name": "Stop", "last_assistant_message": "All done"}))
        event = json.loads((Path(tmp) / "abc.done.json").read_text())
        ask = settings["hooks"]["Notification"][0]["hooks"][0]["command"]
        _run_hook(ask, env, stdin=json.dumps({"message": "Claude needs your permission"}))
        asked = json.loads((Path(tmp) / "abc.needs-input.json").read_text())
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["abc.done.json", "abc.needs-input.json"]
    assert event["kind"] == "done" and event["agent"] == "claude" and event["message"] == "All done"
    assert asked["kind"] == "needs-input" and asked["message"] == "Claude needs your permission"


def test_codex_gets_a_notify_override_for_done_only():
    import os
    import tempfile
    try:
        import tomllib
    except ImportError:  # Python before 3.11 has no tomllib
        return
    assert plan_run("codex", "x").argv == ["codex", "exec", "x"]
    plan = plan_run("codex", "fix it", run_id="abc")
    assert plan.push == "push (done only)"
    assert plan.argv[:5] == ["env", "CONDUCTOR_RUN=abc", "codex", "exec", "-c"] and plan.argv[-1] == "fix it"
    assert plan.argv[5].startswith("notify=[")
    notify = tomllib.loads(plan.argv[5])["notify"]  # valid TOML, so codex can parse it
    assert notify[:2] == ["python3", "-c"] and notify[-2:] == ["codex", "done"]
    assert not any(b in _flags(plan.argv) for b in BANNED)
    with tempfile.TemporaryDirectory() as tmp:  # codex appends its JSON as the last argument
        env = {**os.environ, "CONDUCTOR_RUN": "abc", "CONDUCTOR_EVENTS": tmp}
        payload = json.dumps({"type": "agent-turn-complete", "last-assistant-message": "Fixed."})
        _run_hook([*notify, payload], env)
        event = json.loads((Path(tmp) / "abc.done.json").read_text())
    assert event["kind"] == "done" and event["agent"] == "codex" and event["message"] == "Fixed."


def test_the_hook_script_never_fails_the_agent_or_writes_outside_the_folder():
    import os
    import sys
    import tempfile
    script = ROOT / "notify_hook.py"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "events"
        base = {**os.environ, "CONDUCTOR_EVENTS": str(folder)}
        for env, args, stdin in (
            ({**base}, ["claude", "done"], "{broken"),  # no run id, bad json
            ({**base, "CONDUCTOR_RUN": "../escape"}, ["claude", "done"], "{}"),
            ({**base, "CONDUCTOR_RUN": "abc"}, ["claude", "explode"], "{}"),
        ):
            done = _run_hook([sys.executable, str(script), *args], env, stdin=stdin)
            assert done.returncode == 0 and done.stdout == "", args
        assert not folder.exists() or list(folder.iterdir()) == []
        assert not (Path(tmp) / "escape.done.json").exists()
        long = json.dumps({"message": "x" * 5000 + "THE END"})
        _run_hook([sys.executable, str(script), "claude", "done"], {**base, "CONDUCTOR_RUN": "abc"}, stdin=long)
        event = json.loads((folder / "abc.done.json").read_text())
    assert len(event["message"]) == 1500 and event["message"].endswith("THE END")


def test_cursor_recipe_has_no_bypass_flag_and_is_polling_only():
    plan = plan_run("cursor", "fix the test")
    assert plan.argv == ["cursor-agent", "--print", "fix the test"]
    assert plan.push == "polling" and plan.mode == "background"
    for flag in ("--force", "-f", "--yolo", "--trust", "--approve-mcps", "--auto-review"):
        assert flag not in plan.argv
    assert plan_run("cursor-agent", "x").argv[0] == "cursor-agent"
    assert plan_run("cursor", "x", run_id="abc").argv == ["cursor-agent", "--print", "x"]  # nothing to wire
    try:
        plan_run("cursor", "--yolo do it")
    except PlanError:
        pass
    else:
        raise AssertionError("a prompt could smuggle in a flag")


def test_run_labels_how_each_agent_reports_back():
    import tempfile
    spawned = []
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp), spawn=lambda argv, cwd, log: spawned.append(argv) or 0,
                          capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""})
        labels = {a: captain.run(a, "x")["events"] for a in ("claude", "codex", "cursor", "grok", "hermes", "t3")}
        record = captain.run("claude", "x")
    assert labels == {"claude": "push", "codex": "push (done only)", "cursor": "polling",
                      "grok": "polling", "hermes": "polling", "t3": "polling"}
    assert spawned[0][:2] == ["env", f"CONDUCTOR_RUN={spawned[0][1].split('=')[1]}"]
    assert f"CONDUCTOR_RUN={record['id']}" in spawned[-1]  # the hook's run id is the run's id


STOP = "If you are blocked or the goal is unclear, stop and say exactly what you need. Do not guess and do not widen the scope."


def test_verify_command_is_run_and_reported_before_done():
    from hermes_captain.plan import build_prompt
    prompt = build_prompt("fix login", ["a.py"], "the redirect works", verify="pytest -q tests/test_login.py")
    assert "Verify: run `pytest -q tests/test_login.py`. Fix the work until it passes, then report its real output." in prompt
    assert prompt.index("Verify:") < prompt.index("Done when:")
    none = build_prompt("fix login", ["a.py"])
    assert "find the project's test command first, run it" in none and "report its real output" in none
    assert "`" not in none


def test_a_tiny_task_gets_a_tiny_prompt():
    from hermes_captain.plan import compose_prompt
    prompt, parts = compose_prompt("fix the typo on line 3", ["README.md"], "the typo is gone", verify="grep -c teh README.md", agent="claude")
    assert len(prompt.splitlines()) <= 10 and len(prompt) < 650, prompt
    assert "Read the relevant code first" not in prompt and "explore" not in parts
    for padding in ("Context:", "Never touch", "skills", "subagents"):
        assert padding not in prompt
    assert parts == ["rules", "stop", "final_answer", "goal", "files", "verify", "done_when"]


def test_empty_files_or_a_long_goal_adds_the_explore_line():
    from hermes_captain.plan import compose_prompt
    line = "Read the relevant code first and state your plan in 3 lines before editing."
    prompt, parts = compose_prompt("add rate limiting to the api")
    assert line in prompt and "explore" in parts
    assert prompt.rstrip().endswith(line)  # last, right before the agent starts editing
    assert line in compose_prompt("x" * 301, ["a.py"])[0]
    assert line not in compose_prompt("x" * 300, ["a.py"])[0]
    assert line not in compose_prompt("add rate limiting", ["api.py"])[0]


def test_context_is_short_and_avoid_is_a_plain_list():
    from hermes_captain.plan import build_prompt
    prompt = build_prompt("fix login", ["app/login.py"], context="line one\nline two\nline three\nline four", avoid=["migrations/", ".env"])
    assert "Context: line one\nline two\nline three\n" in prompt and "line four" not in prompt
    assert prompt.index("Context:") < prompt.index("Files in scope:")
    assert "Never touch: migrations/, .env." in prompt
    assert "Never touch" not in build_prompt("fix login", ["a.py"])
    assert "Never touch: .env." in build_prompt("fix login", ["a.py"], avoid=".env")


def test_no_shouting_in_the_prompt():
    import re
    from hermes_captain.plan import build_prompt
    prompt = build_prompt("fix login", ["a.py"], "works", verify="pytest", context="why", avoid=["x"], skills=["tdd"], agent="claude")
    assert not re.search(r"\b[A-Z]{3,}\b", prompt) and "!" not in prompt and "important" not in prompt.lower()


def test_skills_line_is_only_for_claude_and_codex_and_only_when_given():
    from hermes_captain.plan import build_prompt
    claude = build_prompt("fix login", ["a.py"], skills=["tdd", "review"], agent="claude")
    assert "Use these skills if they are installed: tdd, review." in claude
    assert "If pieces are independent, use parallel subagents." in claude
    assert "Use these skills" in build_prompt("fix login", ["a.py"], skills="tdd", agent="Claude-Code")
    codex = build_prompt("fix login", ["a.py"], skills=["tdd"], agent="codex")
    assert "Follow AGENTS.md. Apply these skills if it lists them: tdd." in codex and "subagents" not in codex
    for agent in ("claude", "codex"):  # nothing given, nothing said
        quiet = build_prompt("fix login", ["a.py"], agent=agent)
        assert not any(w in quiet.lower() for w in ("skill", "subagent", "agents.md"))
    for agent in ("cursor", "hermes", "grok", "opencode", "t3", "mystery-agent", ""):
        other = build_prompt("fix login", ["a.py"], skills=["tdd", "review"], agent=agent)
        assert not any(w in other.lower() for w in ("skill", "subagent", "agents.md", "tdd")), agent


def test_the_stop_condition_is_always_there():
    from hermes_captain.plan import build_prompt
    for kwargs in (dict(), dict(files=["a.py"]), dict(files=["a.py"], verify="make", skills=["x"], agent="claude"), dict(agent="codex", avoid=["y"])):
        assert STOP in build_prompt("fix login", **kwargs), kwargs
    assert STOP not in build_prompt("fix login", delta=True)  # a continue already has it


def test_stable_text_comes_first_and_the_task_last():
    from hermes_captain.plan import build_prompt
    a = build_prompt("fix login", ["a.py"], "works", verify="pytest", context="old code", avoid=["x"])
    b = build_prompt("add export button", ["b.py", "c.py"], "button shows", verify="npm test", context="new ui")
    rules, stop, final = a.index("One rule"), a.index("If you are blocked"), a.index("End with a final answer")
    assert rules < stop < final < a.index("Context:") < a.index("Goal:") < a.index("Files in scope:") < a.index("Verify:") < a.index("Done when:")
    assert a.split("\n\n")[0] == b.split("\n\n")[0]  # the same opening for every task, so the cache can be reused
    assert "fix login" not in a.split("\n\n")[0]


def test_a_continue_sends_only_the_delta_and_a_steer_is_passed_through():
    import tempfile
    from hermes_captain.plan import compose_prompt
    full, full_parts = compose_prompt("now handle logout", ["a.py"], verify="pytest", agent="claude", skills=["tdd"])
    delta, delta_parts = compose_prompt("now handle logout", ["a.py"], verify="pytest", agent="claude", skills=["tdd"], delta=True)
    assert "One rule" in full and "rules" in full_parts
    for repeated in ("One rule", "No new dependencies", "If you are blocked", "final answer", "skills", "Read the relevant code"):
        assert repeated not in delta, repeated
    assert "Goal: now handle logout" in delta and "Verify: run `pytest`" in delta
    assert not {"rules", "stop", "final_answer", "skills"} & set(delta_parts)
    spawned = []
    entry = _entry()
    with tempfile.TemporaryDirectory() as tmp:
        entry._CAPTAIN = Captain(Path(tmp), spawn=lambda argv, cwd, log: spawned.append(argv) or 0)
        entry._run({"agent": "claude", "goal": "first task", "files": ["a.py"]})
        entry._run({"agent": "claude", "continue": True, "goal": "now handle logout", "files": ["a.py"]})
    assert "One rule" in spawned[0][-1] and "--continue" in spawned[1]
    assert spawned[1][-1].startswith("Goal: now handle logout") and "One rule" not in spawned[1][-1]
    with tempfile.TemporaryDirectory() as tmp:
        captain, calls = _t3_captain(Path(tmp), [])
        captain.steer("abc", "also cover the empty case")
    sent = [a[-1] for a in calls if "--if-busy" in a[-1]][0]
    assert "also cover the empty case" in sent and "One rule" not in sent and "final answer" not in sent


def test_conductor_run_reports_the_prompt_it_sent():
    import tempfile
    seen = []
    entry = _entry()
    with tempfile.TemporaryDirectory() as tmp:
        entry._CAPTAIN = Captain(Path(tmp), capture=lambda argv, c, t: seen.append(argv[argv.index("--prompt") + 1]) or {"code": 0, "stdout": "{}", "error": ""})
        built = json.loads(entry._run({"agent": "t3", "goal": "fix login", "files": ["a.py"], "verify": "pytest", "avoid": ["x"], "context": "why"}))
        free = json.loads(entry._run({"agent": "t3", "prompt": "just do it"}))
        no_goal = json.loads(entry._run({"agent": "t3", "verify": "pytest"}))
    assert built["prompt"]["lines"] == len(seen[0].splitlines())
    assert built["prompt"]["parts"] == ["title", "rules", "stop", "final_answer", "context", "goal", "files", "avoid", "verify"]
    assert free["prompt"] == {"lines": 1, "parts": ["free-form"]}
    assert no_goal == {"ok": False, "error": "goal is missing"}


def test_run_records_last_active():
    import tempfile
    import time
    with tempfile.TemporaryDirectory() as tmp:
        record = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""}).run("t3", "review")
    assert abs(record["last_active"] - time.time()) < 5


def test_a_continue_never_repeats_the_plan_first_line():
    from hermes_captain.plan import build_prompt
    assert "state your plan" in build_prompt("add rate limiting")
    assert "state your plan" not in build_prompt("add rate limiting", delta=True)
    assert "One rule" not in build_prompt("add rate limiting", delta=True)


def test_a_remote_cli_run_goes_through_tmux_and_keeps_the_exit_code():
    from hermes_captain.plan import tmux_argv
    plan = plan_run("claude", "fix it", host=MAC, cwd="/Users/dev/app", run_id="abc123")
    joined = plan.argv[-1]
    assert plan.argv[0] == "ssh" and "tmux" in joined and "abc123" in joined and "/Users/dev/app" in joined
    assert "tmux" in plan.summary
    for flag in ("--dangerously", "--yolo", "--trust", "bypass"):
        assert flag not in joined
    assert "exit \"$(cat" in joined  # the ssh call ends with the agent's own exit code


def test_local_and_t3_runs_do_not_use_tmux():
    assert "tmux" not in " ".join(plan_run("claude", "x", cwd="/a", run_id="abc123").argv)
    assert "tmux" not in " ".join(plan_run("t3", "x", host=MAC, cwd="/a", run_id="abc123").argv)
    assert "tmux" not in " ".join(plan_run("claude", "x", host=MAC, cwd="/a").argv)  # no run id: old path


def test_tmux_refuses_a_run_id_that_is_not_plain():
    from hermes_captain.plan import tmux_argv
    try:
        tmux_argv(MAC, "/a", ["claude"], "a;rm -rf /")
    except PlanError:
        return
    raise AssertionError("a bad run id must be refused")


def test_any_unknown_cli_gets_tmux_too():
    plan = plan_run("kimi", "x", host=MAC, cwd="/a", run_id="abc123")
    assert "tmux" in plan.argv[-1]
    plan = plan_run("mytool", "x", host=MAC, cwd="/a", run_id="abc123", prompt_flag="-p")
    assert "tmux" in plan.argv[-1] and "mytool" in plan.argv[-1]


def test_orca_terminal_wait_uses_the_documented_flags():
    from hermes_captain.plan import plan_orca_read, plan_orca_wait
    handle = "term_abc123"
    plan = plan_orca_wait(handle, timeout_ms=60000)
    assert plan.argv == [
        "orca", "terminal", "wait", "--terminal", handle,
        "--for", "tui-idle", "--timeout-ms", "60000", "--json",
    ]
    assert plan.mode == "capture" and "no model" in plan.summary
    assert not any(flag in _flags(plan.argv) for flag in BANNED)
    exited = plan_orca_wait(handle, condition="exit", timeout_ms=5000)
    assert exited.argv[exited.argv.index("--for") + 1] == "exit"
    assert exited.argv[exited.argv.index("--timeout-ms") + 1] == "5000"
    read = plan_orca_read(handle)
    assert read.argv == ["orca", "terminal", "read", "--terminal", handle, "--json"]
    remote = plan_orca_wait(handle, timeout_ms=60000, host=MAC, environment="local")
    assert remote.argv[0] == "ssh" and MAC in remote.argv
    joined = remote.argv[-1]
    for flag in ("terminal", "wait", "--terminal", handle, "--for", "tui-idle", "--timeout-ms", "60000", "--json", "--environment", "local"):
        assert flag in joined, flag
    try:
        plan_orca_wait(handle, condition="chat")
    except PlanError:
        pass
    else:
        raise AssertionError("an invented wait condition was accepted")
    try:
        plan_orca_wait("term;rm", timeout_ms=1000)
    except PlanError:
        return
    raise AssertionError("a bad terminal handle was accepted")


def _orca_launch(kind, handle):
    return json.dumps({"ok": True, "result": {"outcome": {"kind": kind, "handle": handle}, "receipt": {"detail": kind}}})


def _orca_read(text):
    return json.dumps({"ok": True, "result": {"terminal": {"handle": "term_abc123", "status": "running", "tail": [text]}}})


def _orca_wait_json(satisfied, **extra):
    wait = {"handle": "term_abc123", "condition": "tui-idle", "satisfied": satisfied, "status": "running", "exitCode": None}
    wait.update(extra)
    return {"code": 0 if satisfied else 1, "stdout": json.dumps({"ok": True, "result": {"wait": wait}}), "stderr": ""}


def test_orca_terminal_launch_waits_with_no_model_and_reads_the_answer():
    import tempfile
    from hermes_captain.plan import plan_orca_read, plan_orca_wait
    from hermes_captain.runner import ORCA_POLL_MS
    calls = []
    waits = iter([_orca_wait_json(False), _orca_wait_json(True)])

    def capture(argv, cwd, timeout):
        calls.append(argv)
        if "launch" in argv:
            return {"code": 0, "stdout": _orca_launch("terminal", "term_abc123"), "error": ""}
        if "wait" in argv:
            return next(waits)
        if "read" in argv:
            return {"code": 0, "stdout": _orca_read("Fixed the bug. 3 tests pass."), "error": ""}
        return {"code": 0, "stdout": "", "error": ""}

    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain = Captain(root, spawn=lambda *a: (_ for _ in ()).throw(AssertionError("spawned")), capture=capture)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        record = captain.run("orca", "write the design doc", worker="codex")
        assert record["status"] == "launched" and record["watched"] is True and record["outcome"] == "terminal"
        done = captain.wait_done(record["id"], timeout=8)
        saved = json.loads((root / f"{record['id']}.json").read_text())
    assert done["model"] is False and done["ping"] == "done"
    assert len(sent) == 1 and "finished" in sent[0] and "Fixed the bug. 3 tests pass." in sent[0]
    assert '"ok"' not in sent[0] and "conductor_steer on run" not in sent[0]
    waits_argv = [a for a in calls if "wait" in a]
    assert waits_argv == [plan_orca_wait("term_abc123", timeout_ms=ORCA_POLL_MS).argv, plan_orca_wait("term_abc123", timeout_ms=ORCA_POLL_MS).argv]
    assert [a for a in calls if "read" in a] == [plan_orca_read("term_abc123").argv]
    assert saved["status"] == "exited"
    assert not any(flag in _flags(a) for a in calls for flag in BANNED)


def test_orca_blocked_terminal_is_stuck_not_finished():
    import tempfile
    calls = []

    def capture(argv, cwd, timeout):
        calls.append(argv)
        if "launch" in argv:
            return {"code": 0, "stdout": _orca_launch("terminal", "term_abc123"), "error": ""}
        if "wait" in argv:
            return _orca_wait_json(True, blockedReason="agent-trust-workspace")
        if "read" in argv:
            return {"code": 0, "stdout": _orca_read("Trust this workspace?"), "error": ""}
        return {"code": 0, "stdout": "", "error": ""}

    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp), spawn=lambda *a: (_ for _ in ()).throw(AssertionError("spawned")), capture=capture)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        record = captain.run("orca", "write the design doc")
        done = captain.wait_done(record["id"], timeout=8)
    assert done["ping"] == "needs-attention"
    assert len(sent) == 1 and "waiting on you" in sent[0] and "finished" not in sent[0]
    assert "Trust this workspace?" in sent[0]
    assert len([a for a in calls if "wait" in a]) == 1


def test_orca_chat_launch_is_not_watched_or_finished():
    import tempfile
    calls = []

    def capture(argv, cwd, timeout):
        calls.append(argv)
        return {"code": 0, "stdout": _orca_launch("structured", "sess_1"), "error": ""}

    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain = Captain(root, spawn=lambda *a: (_ for _ in ()).throw(AssertionError("spawned")), capture=capture)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        record = captain.run("orca", "write the design doc", worker="claude")
        done = captain.wait_done(record["id"], timeout=2)
        captain.resume()
        saved = json.loads((root / f"{record['id']}.json").read_text())
    assert record["status"] == "launched" and record["outcome"] == "structured" and record["watched"] is False
    assert record["handle"] == "sess_1"
    assert "not watched" in record["summary"]
    assert done["ping"] is None and done["ok"] is False
    assert saved["status"] == "launched" and "ping" not in saved and "pinged_at" not in saved
    assert sent == []
    assert calls and calls[0][1:3] == ["agent", "launch"]
    assert not any("wait" in a or "read" in a for a in calls)


def test_the_hooks_answer_beats_a_log_full_of_warnings():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        c = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": "", "error": ""})
        log = Path(tmp) / "r.log"
        log.write_text("WARNING noisy settings line\nthe real answer", encoding="utf-8")
        record = {"id": "r1", "agent": "claude", "log": str(log), "event": "done", "event_message": "the real answer", "follow_mode": "advise"}
        message = c._finish_message(record, "done")
    assert "the real answer" in message and "noisy" not in message


def test_a_steered_turn_is_watched_again_and_pings_when_it_finishes():
    """Without this a drive follow-up went out and its result never came back."""
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, calls = _t3_captain(root, [COMPLETED, COMPLETED])
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        assert captain.wait_done("abc", timeout=8)["ping"] == "done"
        assert captain.steer("abc", "now add a test")["ok"] is True
        done = captain.wait_done("abc", timeout=8)
        saved = json.loads((root / "abc.json").read_text())
    assert done["ping"] == "done" and len(sent) == 2
    assert saved["rounds"] == 1 and "round 2 of 5" not in sent[0]


def test_a_second_wait_on_the_same_run_does_not_ping_twice():
    import tempfile
    import threading
    sent, gate = [], threading.Event()
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [])
        slow = captain.capture
        captain.capture = lambda argv, cwd, timeout: (gate.wait(5), COMPLETED)[1] if " wait " in argv[-1] else slow(argv, cwd, timeout)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        captain.arm_wait("abc")
        first = captain._waits["abc"]
        assert captain.arm_wait("abc")["waiting"] is True and captain._waits["abc"] is first
        gate.set()
        captain.wait_done("abc", timeout=8)
    assert len(sent) == 1


def test_another_live_process_keeps_its_own_run():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captain, calls = _t3_captain(root, [])
        record = json.loads((root / "abc.json").read_text())
        (root / "abc.json").write_text(json.dumps({**record, "owner": os.getppid()}), encoding="utf-8")
        captain.resume()
        assert "abc" not in captain._waits and calls == []


def test_orca_steer_and_stop_use_the_terminal_and_drive_can_follow_up():
    import tempfile
    from hermes_captain.plan import plan_stop
    assert plan_steer("orca", "term_abc123", "add a test", environment="local").argv == [
        "orca", "terminal", "send", "--terminal", "term_abc123", "--text", "add a test", "--enter", "--json", "--environment", "local",
    ]
    assert plan_stop("orca", "term_abc123").argv == ["orca", "terminal", "send", "--terminal", "term_abc123", "--interrupt", "--json"]
    for bad in (("orca", "term;rm", "x"), ("t3", "th_1", "--permission full-access")):
        try:
            plan_steer(*bad)
        except PlanError:
            continue
        raise AssertionError(f"{bad} was accepted")
    calls, sent = [], []

    def capture(argv, cwd, timeout):
        calls.append(argv)
        if "launch" in argv:
            return {"code": 0, "stdout": _orca_launch("terminal", "term_abc123"), "error": ""}
        if "wait" in argv:
            return _orca_wait_json(True)
        if "read" in argv:
            return {"code": 0, "stdout": _orca_read("Done."), "error": ""}
        return {"code": 0, "stdout": "{}", "error": ""}

    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp), capture=capture)
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        record = captain.run("orca", "write the design doc", mode="drive")
        captain.wait_done(record["id"], timeout=8)
        assert captain.steer(record["id"], "add a test")["ok"] is True
        captain.wait_done(record["id"], timeout=8)
        assert captain.stop(record["id"])["ok"] is True
    assert len(sent) == 2 and f"conductor_steer on run {record['id']} (round 1 of 5)" in sent[0]
    assert any("--interrupt" in a for a in calls)


def test_a_stale_orca_handle_is_lost_not_working_forever():
    """Live Orca 1.4: a timeout and a stale handle both exit 1; only the error code tells them apart."""
    import tempfile
    stale = {"code": 1, "stdout": json.dumps({"ok": False, "error": {"code": "terminal_handle_stale"}}), "stderr": ""}
    timeout = {"code": 1, "stdout": json.dumps({"ok": False, "error": {"code": "timeout"}}), "stderr": ""}
    for replies, ping, waits in (([stale] * 3, "wait failed", 3), ([timeout, _orca_wait_json(True)], "done", 2)):
        calls, feed = [], iter(replies)

        def capture(argv, cwd, timeout):
            calls.append(argv)
            if "wait" in argv:
                return next(feed)
            return {"code": 0, "stdout": _orca_read("ok"), "error": ""}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "abc.json").write_text(json.dumps({
                "id": "abc", "agent": "orca", "surface": "orca", "outcome": "terminal", "handle": "term_abc123", "status": "launched",
            }), encoding="utf-8")
            captain = Captain(root, capture=capture)
            captain.retry_delay = 0
            captain.notify = lambda run_id, record: True
            captain.arm_wait("abc")
            assert captain.wait_done("abc", timeout=8)["ping"] == ping
        assert len([a for a in calls if "wait" in a]) == waits


def test_needs_input_does_not_end_the_watch_on_a_live_agent():
    import tempfile
    import threading
    with tempfile.TemporaryDirectory() as tmp:
        def write(folder):
            (folder / "abc.needs-input.json").write_text(_event("needs-input", "permission for Bash"), encoding="utf-8")
            threading.Timer(1, lambda: (folder / "abc.done.json").write_text(_event("done", "Shipped it."), encoding="utf-8")).start()
        captain, done, sent, alive = _push_wait(Path(tmp), ["sleep", "30"], "push", before_exit=write)
    assert alive and done["ping"] == "done"
    assert len(sent) == 2 and "waiting on you" in sent[0] and "Shipped it." in sent[1]


def test_t3_read_gives_the_answer_not_the_json_wrapper():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [])
        record = captain.read("abc")
        first, second = captain.peek("abc"), captain.peek("abc")
    assert record["output"] == ANSWER and record["status"] == "running"
    assert first["speak"] is True and second["speak"] is False


def test_recipes_take_any_terminal_agent_and_no_prompt_reads_as_a_flag():
    from hermes_captain.plan import RECIPES
    for agent, (words, _) in RECIPES.items():
        plan = plan_run(agent, "fix the test")
        assert plan.argv == [*words, "fix the test"] and plan.push == "polling"
        assert not any(flag in _flags(plan.argv) for flag in BANNED)
    assert plan_run("aider", "fix it", prompt_flag="--message").argv == ["aider", "--message", "fix it"]
    assert plan_run("t3code", "x").surface == "t3"
    assert "--model" in plan_run("orca", "x", model="gpt-6").argv
    for agent, flag in (("claude", None), ("codex", None), ("aider", "--message"), ("aider", "--yes")):
        try:
            plan_run(agent, "--dangerously-skip-permissions" if flag != "--yes" else "x", prompt_flag=flag)
        except PlanError:
            continue
        raise AssertionError(f"{agent} {flag} was accepted")


def test_a_quiet_peek_holds_one_reply_only():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp))
        captain.last_peek["chat-a"] = {"speak": False, "run_id": "abc"}
        assert captain.hold_status_reply("Still running, nothing new.", session="chat-b") is None  # another chat is untouched
        assert captain.hold_status_reply("Still running, nothing new.", session="chat-a") == "No change."
        assert captain.hold_status_reply("There is no update on the deploy yet.", session="chat-a") is None


def test_tools_never_raise_and_a_failed_launch_is_not_ok():
    import tempfile
    spec = importlib.util.spec_from_file_location("hermes_captain.entry", ROOT / "__init__.py")
    entry = importlib.util.module_from_spec(spec)
    entry.__package__ = "hermes_captain"
    spec.loader.exec_module(entry)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        entry._CAPTAIN = Captain(root, capture=lambda *a: {"code": 127, "stdout": "", "error": "not found: t3code"})
        failed = json.loads(entry._run({"agent": "t3", "prompt": "x"}))
        (root / "bad.json").write_text("{not json", encoding="utf-8")
        broken = json.loads(entry._peek({"run_id": "bad"}))
        (root / "abc.json").write_text(json.dumps({"id": "abc", "follow_up": "undelivered", "message": "m"}), encoding="utf-8")
        report = entry.handle_command("")
    assert failed["ok"] is False and "not found: t3code" in failed["error"]
    assert broken["ok"] is False
    assert "allow_gateway_injection" in report


def test_a_just_opened_orca_terminal_is_not_finished_and_a_stop_is_not_a_finish():
    """Live: tui-idle was true 0.4s after launch, before the agent took the prompt."""
    import tempfile
    import time
    sent = []

    def capture(argv, cwd, timeout):
        if "launch" in argv:
            return {"code": 0, "stdout": _orca_launch("terminal", "term_abc123"), "error": ""}
        if "wait" in argv:
            return _orca_wait_json(True)
        return {"code": 0, "stdout": _orca_read("pong"), "error": ""}

    _runner.ORCA_SETTLE = 1
    try:
        with tempfile.TemporaryDirectory() as tmp:
            captain = Captain(Path(tmp), capture=capture)
            captain.retry_delay = 0.05
            captain.notify = lambda run_id, record: sent.append(record["message"]) or True
            started = time.time()
            record = captain.run("orca", "x")
            captain.wait_done(record["id"], timeout=8)
            assert len(sent) == 1 and time.time() - started >= 1
    finally:
        _runner.ORCA_SETTLE = 0
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [COMPLETED])
        captain.notify = lambda run_id, record: sent.append(record["message"]) or True
        assert captain.stop("abc")["ok"] is True
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
        captain.capture = lambda *a: {"code": 4, "stdout": "", "error": "no running turn to interrupt"}
        assert captain.stop("abc")["ok"] is False
        assert json.loads((Path(tmp) / "abc.json").read_text())["status"] == "stopped"
    assert "was stopped" in sent[1] and "finished" not in sent[1]



def test_lead_kickoff_makes_the_agent_delegate_and_is_not_resent():
    from hermes_captain.plan import LEAD, compose_prompt
    prompt, parts = compose_prompt("add gift cards", ["src/cart"], "tests pass", verify="npm test", lead=True)
    assert LEAD in prompt and "lead" in parts
    assert prompt.index(LEAD) < prompt.index("Goal:")
    assert LEAD not in compose_prompt("add gift cards", ["src/cart"])[0]
    assert LEAD not in compose_prompt("now the emails", lead=True, delta=True)[0]


def test_a_day_old_undelivered_message_is_not_retried():
    import tempfile
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [COMPLETED])
        captain.notify = lambda run_id, record: False
        captain.arm_wait("abc")
        captain.wait_done("abc", timeout=8)
        path = Path(tmp) / "abc.json"
        record = json.loads(path.read_text())
        record["pinged_at"] -= 2 * 86400
        path.write_text(json.dumps(record))
        again = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": ""})
        again.notify = lambda run_id, record: sent.append(1) or True
        again.resume()
    assert not sent


def test_t3_approval_pings_once_then_the_finish_still_arrives():
    import tempfile
    needs = {"code": 0, "stdout": json.dumps({"data": {"wait": {"outcome": "needs-attention"}}}), "stderr": ""}
    sent = []
    with tempfile.TemporaryDirectory() as tmp:
        captain, _ = _t3_captain(Path(tmp), [needs, needs, needs, COMPLETED])
        captain.notify = lambda run_id, record: sent.append(record["ping"]) or True
        captain.arm_wait("abc")
        captain._waits["abc"].join(8)
    assert sent == ["needs-attention", "done"]


def test_t3_thread_is_named_after_the_goal():
    from hermes_captain.plan import compose_prompt
    prompt, parts = compose_prompt("Fix the failing checkout test", ["src/cart"], agent="t3")
    assert prompt.splitlines()[0] == "Fix the failing checkout test" and parts[0] == "title"
    assert compose_prompt("Fix it", agent="claude")[0].startswith("One rule")
    assert compose_prompt("Fix it", agent="t3", delta=True)[1][0] != "title"


def test_twelve_agents_at_once_each_ping_once():
    import time
    import tempfile
    sent = {}
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp))
        captain.notify = lambda run_id, record: sent.setdefault(run_id, []).append(record["ping"]) or True
        runs = [captain.run("echo", f"agent {i} done", prompt_flag="-q") for i in range(12)]
        end = time.time() + 30
        while len(sent) < 12 and time.time() < end:
            time.sleep(0.1)
    assert len({r["id"] for r in runs}) == 12
    assert sorted(sent) == sorted(r["id"] for r in runs)
    assert all(pings == ["done"] for pings in sent.values())


def _owner_refuses(call):
    try:
        call()
    except PlanError as exc:
        assert "owner" in str(exc)
        return
    raise AssertionError("was not refused")


def test_hosts_and_extra_commands_need_the_owners_allowlist():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp), capture=lambda *a: {"code": 0, "stdout": "{}", "error": ""}, spawn=lambda *a: 4242)
        captain.allowed_hosts, captain.allowed_commands = set(), set()  # what the plugin sets by default
        for call in (
            lambda: captain.run("node", "process.exit()", prompt_flag="-p"),
            lambda: captain.run("python3", "http.server", prompt_flag="-m"),
            lambda: captain.run("claude", "fix it", host="dev@box"),
            lambda: captain.run("orca", "fix it", worker="claude", orca_host="box"),
            lambda: captain.doctor("dev@box"),
            lambda: captain.watch_running("dev@box"),
        ):
            _owner_refuses(call)
        assert not list(Path(tmp).glob("*.json"))  # nothing was started
        captain.allowed_hosts, captain.allowed_commands = {"dev@box"}, {"myagent"}
        assert captain.run("myagent", "fix it", prompt_flag="-p")["status"] == "running"
        assert captain.run("claude", "fix it")["status"] == "running"  # a built-in agent on this machine needs no listing
        assert captain.doctor("dev@box")["host"] == "dev@box"


def test_plugin_reads_allowlists_from_owner_settings_and_asks_approval():
    entry = _entry()

    class Ctx:
        def get_config(self, key, default=None):
            return {"hosts": "dev@box, mac", "commands": ["myagent"]}.get(key, default)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        entry._CAPTAIN, entry._CTX = Captain(Path(tmp)), Ctx()
        try:
            entry._apply_settings()
            assert entry._CAPTAIN.allowed_hosts == {"dev@box", "mac"} and entry._CAPTAIN.allowed_commands == {"myagent"}
            entry._CTX = type("Broken", (), {"get_config": lambda self, *a: 1 / 0})()
            entry._apply_settings()
            assert entry._CAPTAIN.allowed_hosts == set() and entry._CAPTAIN.allowed_commands == set()
        finally:
            entry._CTX = None
    run = entry._approve("conductor_run", {"agent": "claude", "cwd": "/srv/app", "host": "mac", "goal": "Fix login\nmore"})
    assert run["action"] == "approve" and run["message"] == "claude in /srv/app on mac: Fix login" and "claude:mac:/srv/app" in run["rule_key"]
    steer = entry._approve("conductor_steer", {"run_id": "abc", "prompt": "add a test"})
    assert steer["action"] == "approve" and "abc" in steer["message"] and "add a test" in steer["message"]
    assert entry._approve("conductor_peek", {"run_id": "abc"}) is None
    assert entry._approve("terminal", {"command": "ls"}) is None


def test_children_get_a_minimal_environment(monkeypatch):
    from hermes_captain.runner import _child_env
    for key, value in {"OPENAI_API_KEY": "sk-o", "ANTHROPIC_API_KEY": "sk-a", "XAI_API_KEY": "x", "TELEGRAM_BOT_TOKEN": "t",
                       "SSH_AUTH_SOCK": "/tmp/s", "PATH": "/usr/bin", "HOME": "/home/u", "CONDUCTOR_RUN": "abc"}.items():
        monkeypatch.setenv(key, value)
    ssh = _child_env(["ssh", "dev@box", "true"])
    assert ssh["PATH"] == "/usr/bin" and ssh["HOME"] == "/home/u" and ssh["CONDUCTOR_RUN"] == "abc" and ssh["SSH_AUTH_SOCK"] == "/tmp/s"
    assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "TELEGRAM_BOT_TOKEN"} & set(ssh)
    claude = _child_env(["/opt/bin/claude", "-p", "x"])
    assert claude["ANTHROPIC_API_KEY"] == "sk-a" and "OPENAI_API_KEY" not in claude and "TELEGRAM_BOT_TOKEN" not in claude
    assert "OPENAI_API_KEY" in _child_env(["codex", "exec", "x"]) and "ANTHROPIC_API_KEY" not in _child_env(["codex", "exec", "x"])
    assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SSH_AUTH_SOCK"} & set(_child_env(["echo", "hi"]))


def test_session_key_comes_from_the_public_hermes_call(monkeypatch):
    import types
    entry = _entry()
    fake = types.ModuleType("gateway.session_context")
    fake.get_session_env = lambda name, default="": "agent:main:telegram:dm:1" if name == "HERMES_SESSION_KEY" else default
    monkeypatch.setitem(sys.modules, "gateway", types.ModuleType("gateway"))
    monkeypatch.setitem(sys.modules, "gateway.session_context", fake)
    assert entry._session({"session_key": "fallback"}) == "agent:main:telegram:dm:1"
    assert "bound_session_key" not in (ROOT / "__init__.py").read_text()


def test_a_saved_run_on_an_unlisted_host_is_not_reached():
    """read, peek, wait, steer, stop and the restart resume take a run id, not a host: the list applies to them too."""
    import tempfile
    calls = []

    def capture(argv, cwd, timeout):
        calls.append(argv)
        return {"code": 0, "stdout": "{}", "stderr": "", "error": ""}

    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp), capture=capture)
        base = {"status": "launched", "last_active": 0, "cwd": "/srv/app"}
        captain._save("t3run", {**base, "id": "t3run", "agent": "t3", "surface": "t3", "handle": "th_1", "host": "old@box"})
        captain._save("orcarun", {**base, "id": "orcarun", "agent": "orca", "surface": "orca", "outcome": "terminal",
                                  "handle": "term_1", "watched": True, "environment": "oldbox"})
        captain._save("clirun", {**base, "id": "clirun", "agent": "claude", "surface": "cli", "pid": 4242, "host": "old@box"})
        captain.allowed_hosts = set()
        for run_id in ("t3run", "orcarun", "clirun"):
            for call in (captain.read, captain.peek, captain.arm_wait, captain.stop, lambda r: captain.steer(r, "go on")):
                _owner_refuses(lambda: call(run_id))
        captain.resume()  # a restart does not reach them either, and does not stop at the first one
        assert calls == [] and not captain._waits
        captain.allowed_hosts = {"old@box"}
        assert captain.read("t3run")["id"] == "t3run" and calls and calls[0][0] == "ssh"


def test_settings_are_applied_before_the_first_resume():
    entry = _entry()
    seen = []

    class Ctx:
        def get_config(self, key, default=None):
            return default

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        captain = Captain(Path(tmp))
        captain.resume = lambda: seen.append(captain.allowed_hosts)
        entry._CAPTAIN = captain
        try:
            entry.register(Ctx())
        finally:
            captain._supervise_stop.set()
            entry._CTX = None
    assert seen and all(hosts == set() for hosts in seen)


def test_the_approval_prompt_names_the_orca_machine():
    ask = _entry()._approve("conductor_run", {"agent": "orca", "cwd": "/srv/app", "orca_host": "buildbox", "goal": "Fix login"})
    assert ask["message"] == "orca in /srv/app on buildbox: Fix login" and "orca:buildbox:/srv/app" in ask["rule_key"]


def test_every_subprocess_gets_the_minimal_environment(monkeypatch):
    import ast
    import os
    import subprocess
    import tempfile
    from hermes_captain.peek import git_status
    from hermes_captain.runner import _child_env, _proc_start
    for path in ROOT.glob("*.py"):  # no call may inherit what Hermes loaded
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("subprocess.run", "subprocess.Popen", "subprocess.check_output"):
                assert any(k.arg == "env" for k in node.keywords), f"{path.name}:{node.lineno} has no env="
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    envs = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: envs.append((argv[0], kw.get("env"))) or real(argv, **kw))
    with tempfile.TemporaryDirectory() as tmp:
        git_status(tmp)
    _proc_start(os.getpid())
    assert [name for name, _ in envs] == ["git", "ps"]
    assert all(env is not None and "PATH" in env and not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"} & set(env) for _, env in envs)
    # the real launch puts `env CONDUCTOR_RUN=id` first: the agent behind it still gets its own key, and only its own
    claude = _child_env(plan_run("claude", "fix it", run_id="abc").argv)
    codex = _child_env(plan_run("codex", "fix it", run_id="abc").argv)
    assert claude["ANTHROPIC_API_KEY"] == "sk-a" and "OPENAI_API_KEY" not in claude
    assert codex["OPENAI_API_KEY"] == "sk-o" and "ANTHROPIC_API_KEY" not in codex


if __name__ == "__main__":  # last in the file, so it runs every test above
    import inspect
    import os
    import traceback
    from unittest import mock

    class MonkeyPatch:  # the three calls the tests use, so this file runs without pytest
        def __init__(self):
            self._patches = []

        def _start(self, patch):
            self._patches.append(patch)
            patch.start()

        def setenv(self, key, value):
            self._start(mock.patch.dict(os.environ, {key: value}))

        def setitem(self, mapping, key, value):
            self._start(mock.patch.dict(mapping, {key: value}))

        def setattr(self, target, name, value):
            self._start(mock.patch.object(target, name, value))

        def undo(self):
            while self._patches:
                self._patches.pop().stop()

    failed = 0
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        patch = MonkeyPatch()
        try:
            test(patch) if "monkeypatch" in inspect.signature(test).parameters else test()
            print("ok", test.__name__)
        except Exception:
            failed += 1
            print("FAIL", test.__name__)
            traceback.print_exc()
        finally:
            patch.undo()
    if failed:
        raise SystemExit(f"{failed} failed")
    print(f"{len(tests)} passed")

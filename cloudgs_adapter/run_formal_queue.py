"""Persistent, single-GPU PRTGaussian CloudDataset 01/02 training queue.

Run this process detached using the project's venv. It holds the project lock
until both cases finish. Polling reads files and process metadata only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MIN_FREE_BYTES = 10 * 1024 ** 3
SOURCES = (
    ("01", "/myfiles/data/CloudDatasetUniform"),
    ("02", "/myfiles/data/CloudDatasetUniform_02"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a read-only existence check on Windows.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie cannot make progress even though kill(pid, 0) succeeds.
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        if any(line.startswith("State:") and "Z" in line for line in status.splitlines()):
            return False
    except (FileNotFoundError, PermissionError):
        pass
    return True


def log_tail(path, count=3):
    if not path or not Path(path).is_file():
        return []
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 65536))
        text = stream.read().decode("utf-8", errors="replace")
    return [line[-2000:] for line in text.splitlines()[-count:]]


def progress(output):
    if not output:
        return None
    for line in reversed(log_tail(Path(output) / "traininglog.jsonl", count=8)):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "stage" in row and "step" in row:
            return row
    return None


def poll(control):
    path = Path(control) / "state.json"
    if not path.is_file():
        return {"status": "not_started", "control": str(control)}
    state = read_json(path)
    result = {key: state.get(key) for key in
              ("status", "pid", "started", "updated", "finished", "error")}
    result["pid_alive"] = pid_alive(state.get("pid"))
    result["cases"] = []
    for case in state.get("cases", []):
        item = dict(case)
        item["pid_alive"] = pid_alive(case.get("pid"))
        item["progress"] = progress(case.get("output"))
        item["tail"] = log_tail(case.get("log"))
        result["cases"].append(item)
    return result


@contextmanager
def project_lock(repo):
    # Import lazily so read-only --poll and CPU logic checks also work on Windows.
    import fcntl
    path = repo / ".experiment-control" / "prt_training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another PRTGaussian queue holds the project training lock") from error
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "started": now()}))
        stream.flush()
        # Closing this descriptor releases the lock only after any inherited
        # training-child descriptor also closes; explicit LOCK_UN would not.
        yield stream.fileno()


def require_idle_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=20, check=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if rows:
        raise RuntimeError("GPU already has compute processes; refusing to start: " + "; ".join(rows))


def require_free_disk(repo):
    free = shutil.disk_usage(repo).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"Only {free / 1024 ** 3:.2f} GiB free; at least 10 GiB required")
    return free


def build_plan(repo, resume_control=None):
    plan = {
        "repo": str(repo), "stage1_iterations": 30000,
        "stage2_iterations": 30000, "resolution": 1024, "batch_size": 8,
        "preview_every": 1000, "checkpoint_every": 7000,
        "network_config": "config/config.json",
        "cases": [{"dataset": identifier, "source": source} for identifier, source in SOURCES],
    }
    if resume_control is None:
        return plan
    resume_control = Path(resume_control).absolute()
    previous_plan = read_json(resume_control / "plan.json")
    previous = read_json(resume_control / "state.json")
    for key in ("repo", "stage1_iterations", "stage2_iterations", "resolution", "batch_size",
                "preview_every", "checkpoint_every", "network_config"):
        if previous_plan.get(key) != plan[key]:
            raise RuntimeError(f"Originating queue has a different {key}; refusing to change protocol")
    if [(case.get("dataset"), case.get("source")) for case in previous_plan.get("cases", [])] != list(SOURCES):
        raise RuntimeError("Originating queue has different datasets or ordering")
    if pid_alive(previous.get("pid")) or any(pid_alive(case.get("pid")) for case in previous.get("cases", [])):
        raise RuntimeError("Originating queue or training child is still alive; stop it before resuming")
    previous_cases = previous.get("cases", [])
    if [(case.get("dataset"), case.get("source")) for case in previous_cases] != list(SOURCES):
        raise RuntimeError("Originating queue state has different datasets or ordering")
    plan["resume_control"] = str(resume_control)
    for case, origin in zip(plan["cases"], previous_cases):
        if not origin.get("output"):
            if origin.get("status") != "pending":
                raise RuntimeError(f"Dataset {case['dataset']} has no resumable output")
            continue
        output = Path(origin["output"]).absolute()
        try:
            output.resolve().relative_to((repo / "output").resolve())
        except ValueError as error:
            raise RuntimeError("Originating output is outside this project's output directory") from error
        summary_path = output / "validation_summary.json"
        if summary_path.is_file() and read_json(summary_path).get("status") == "complete":
            validate_success(output)
            case["reuse_output"] = str(output)
        else:
            coarse, checkpoint = output / "coarse_xyz.pt", output / "checkpoint_latest.pt"
            for path in (coarse, checkpoint):
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Dataset {case['dataset']} has no complete resume file: {path}")
            case.update(stage1_checkpoint=str(coarse), resume_stage2=str(checkpoint),
                        resume_output=str(output))
        case["origin_status"] = origin.get("status")
    return plan


def build_command(repo, case, output):
    # Never resolve() this symlink: doing so switches to the system Python.
    python = repo / ".venv" / "bin" / "python"
    command = [str(python), "-u", "-m", "cloudgs_adapter.train_cloud",
            "--source", case["source"], "--output", str(output),
            "--stage1-iterations", "30000", "--stage2-iterations", "30000",
            "--resolution", "1024", "--batch-size", "8",
            "--preview-every", "1000", "--checkpoint-every", "7000",
            "--network-config", "config/config.json"]
    if case.get("resume_stage2"):
        command.extend(["--stage1-checkpoint", case["stage1_checkpoint"],
                        "--resume-stage2", case["resume_stage2"]])
    return command


def validate_success(output):
    summary = read_json(Path(output) / "validation_summary.json")
    if summary.get("status") != "complete":
        raise RuntimeError("Training summary does not report complete")
    if summary.get("diagnostic_only") is not False:
        raise RuntimeError("A diagnostic run cannot satisfy a formal case")
    if summary.get("source_sha256_before_equals_after") is not True:
        raise RuntimeError("Source PLY/split hash preservation was not verified")
    stages = summary.get("stages", {})
    for name in ("stage1", "stage2"):
        if stages.get(name, {}).get("steps") != 30000:
            raise RuntimeError(f"Unexpected {name} iteration budget")
    if summary.get("resolution") != 1024 or stages["stage2"].get("batch_size") != 8:
        raise RuntimeError("Unexpected formal resolution or batch size")
    return summary


def new_output(repo, identifier):
    for _ in range(5):
        output = repo / "output" / (datetime.now().strftime("%Y%m%d_%H%M%S") +
                                    f"_prt_cloud{identifier}_envoff")
        if not output.exists():
            return output
        time.sleep(1.05)
    raise RuntimeError("Could not allocate a fresh timestamp output directory")


def run(control, repo, resume_control=None):
    if resume_control is not None and Path(resume_control).resolve() == control.resolve():
        raise RuntimeError("A resumed queue requires a new control directory; origin state is preserved")
    control.mkdir(parents=True, exist_ok=True)
    with project_lock(repo) as lock_fd:
        plan = build_plan(repo, resume_control)
        plan_path, state_path = control / "plan.json", control / "state.json"
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise RuntimeError("Existing queue plan differs; refusing to overwrite it")
        else:
            atomic_json(plan_path, plan)
        if state_path.exists():
            previous = read_json(state_path)
            if previous.get("status") == "finished":
                for case in previous["cases"]:
                    validate_success(case["output"])
                return previous
            raise RuntimeError("Existing unfinished/failed queue requires operator review; no automatic retry")
        python = repo / ".venv" / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"Project venv Python not found: {python}")
        for case in plan["cases"]:
            for filename in ("points3d.ply", "transforms_train.json", "transforms_test.json"):
                if not (Path(case["source"]) / filename).is_file():
                    raise FileNotFoundError(Path(case["source"]) / filename)
        require_idle_gpu()
        free = require_free_disk(repo)
        state = {"status": "running", "pid": os.getpid(), "started": now(),
                 "updated": now(), "initial_free_bytes": free,
                 "cases": [dict(case, status="pending", pid=None) for case in plan["cases"]]}
        atomic_json(state_path, state)
        active = None
        try:
            for case in state["cases"]:
                active = case
                if case.get("reuse_output"):
                    summary = validate_success(case["reuse_output"])
                    case.update(status="complete", output=case["reuse_output"],
                                reused_completed=True, stages=summary["stages"], finished=now(),
                                checkpoint_restore_max_abs_error=summary.get("checkpoint_restore_max_abs_error"))
                    state["updated"] = now()
                    atomic_json(state_path, state)
                    continue
                require_idle_gpu()
                case["free_bytes_before"] = require_free_disk(repo)
                output = new_output(repo, case["dataset"])
                log = control / "logs" / (output.name + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                command = build_command(repo, case, output)
                case.update(status="starting", output=str(output), log=str(log),
                            args=command, started=now())
                state["updated"] = now()
                atomic_json(state_path, state)
                # No shell and no daemonizing here; parent detaches this queue.
                with log.open("xb") as stream:
                    process = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               close_fds=True, pass_fds=(lock_fd,))
                    case.update(status="running", pid=process.pid)
                    state["updated"] = now()
                    atomic_json(state_path, state)
                    returncode = process.wait()
                case.update(returncode=returncode, finished=now())
                if returncode:
                    raise RuntimeError(f"Dataset {case['dataset']} training exited {returncode}; output preserved")
                summary = validate_success(output)
                case.update(status="complete", stages=summary["stages"],
                            checkpoint_restore_max_abs_error=summary.get("checkpoint_restore_max_abs_error"))
                state["updated"] = now()
                atomic_json(state_path, state)
                time.sleep(2)
            state.update(status="finished", finished=now(), updated=now())
            atomic_json(state_path, state)
            return state
        except BaseException as error:
            if active is not None and active.get("status") != "complete":
                active.update(status="failed", error=str(error), finished=now())
            state.update(status="failed", error=str(error), finished=now(), updated=now())
            atomic_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).absolute().parent.parent)
    parser.add_argument("--resume-control", type=Path,
                        help="Stopped originating queue; keep completed cases and resume Stage 2 checkpoints")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--poll", action="store_true")
    args = parser.parse_args()
    if not args.control.is_absolute():
        parser.error("--control must be absolute")
    if args.resume_control is not None and (not args.run or not args.resume_control.is_absolute()):
        parser.error("--resume-control requires --run and an absolute originating control path")
    repo = args.repo.absolute()
    try:
        result = poll(args.control) if args.poll else run(args.control, repo, args.resume_control)
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

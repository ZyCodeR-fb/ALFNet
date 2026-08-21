#!/usr/bin/env python
"""The v4 development-base launcher: parallel, GPU-bound, resumable, honest.

Written in Python rather than shell because the four things that actually
matter here are all things shell does badly: an atomic per-run lock, a bounded
worker pool with a phase barrier, forwarding a signal to a process *group* so no
orphaned trainer survives the launcher, and a failure that propagates instead of
being swallowed by `|| true`. `run_v4_development_base.sh` is the documented
entry point and execs this.

Four properties, each of which the first launcher lacked:

* **parallel, one job per GPU.** `JOBS` workers, each pinned to one device from
  `DEVICES`. More jobs than devices is refused unless `ALLOW_GPU_SHARING=1` says
  so deliberately, because two trainers sharing a card is a memory decision
  somebody should make on purpose.
* **a health barrier.** The four seed-32 runs go first and must all reach
  COMPLETE before any seed-42/52 run starts. A problem specific to one dataset
  or model surfaces after roughly one run each rather than after every PEMS08
  seed has finished. Their *numbers* decide nothing -- the queue does not branch
  on them, only on whether they completed.
* **real resume.** A run with a valid `last.pt` is continued with `--resume`, at
  its next epoch, not restarted. A directory that is non-empty but has no
  `last.pt` is refused rather than overwritten.
* **failures that are visible, and stops that are not failures.** A run that
  crashes writes `FAILED.json`, stops its phase, and makes the launcher exit
  non-zero. A run the *user* stopped writes no such thing: it keeps its
  `last.pt`, stays RESUMABLE, and the same command resumes it. The two are told
  apart by whether this launcher forwarded the signal, not by the exit code --
  an OOM kill also exits on a signal.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_v4_development_plan import (  # noqa: E402
    assert_plan_still_holds,
    load_plan,
    sha256_file,
)
from scripts.train_v4 import (  # noqa: E402
    COMPLETE_MARKER,
    FAILED_MARKER,
    LAUNCHER_VERSION,
    RUN_STATE_NAME,
    run_artefacts,
)

# Retained only as the development plan's historical shape; nothing enumerates
# runs from these. `plan_runs` reads the plan.
HEALTH_SEED = 32
LOCK_NAME = ".lock"
POLL_SECONDS = 5


class LauncherRefused(RuntimeError):
    """The queue will not start, or will not continue, in this state."""


def resolve_plan_and_root(plan_path=None):
    """Load a frozen plan and the output root it names. Never guesses either."""
    if plan_path is None:
        plan = assert_plan_still_holds(load_plan())
        return plan, ROOT / "outputs_v4/development_base" / plan["plan_sha256"]
    from scripts import build_v4_five_dataset_plan as five
    from scripts.build_v4_readiness import ReadinessRefused, assert_ready

    plan = five.assert_plan_still_holds(five.load_plan(plan_path))
    # The running permit, checked here -- before a run directory exists, before
    # a child is spawned and before anything touches CUDA. A manifest flag that
    # the entry point ignores is worse than no flag.
    if plan.get("readiness_attestation"):
        try:
            assert_ready(plan)
        except ReadinessRefused as error:
            raise LauncherRefused(f"{plan_path}: {error}") from error
    root = plan.get("output_root")
    if not root:
        raise LauncherRefused(f"{plan_path} declares no output_root")
    return plan, ROOT / root / plan["plan_sha256"]


@dataclass
class Run:
    dataset: str
    model: str
    seed: int
    phase: int
    config: Path
    directory: Path
    log: Path
    device: str = None
    process: object = field(default=None, repr=False)
    lock: object = field(default=None, repr=False)
    started: float = None
    # "train" or "reuse". A reused run is an already-verified checkpoint from
    # another plan; it occupies a result slot, never a GPU worker.
    execution_action: str = "train"
    entry: dict = field(default_factory=dict, repr=False)

    @property
    def name(self):
        return f"{self.dataset}/{self.model}/seed{self.seed}"


# --- planning -----------------------------------------------------------------


def parse_devices(devices):
    """`"0,1,2"` -> `["cuda:0", "cuda:1", "cuda:2"]`. `cpu` is allowed for tests."""
    parsed = []
    for token in str(devices).split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(token if token.startswith("cuda:") or token == "cpu"
                      else f"cuda:{int(token)}")
    if not parsed:
        raise LauncherRefused("DEVICES is empty")
    return parsed


def assign_slots(jobs, devices, allow_sharing=False):
    """One device per worker slot. Sharing a card is opt-in, never a default."""
    jobs = int(jobs)
    if jobs < 1:
        raise LauncherRefused(f"JOBS must be at least 1, got {jobs}")
    unique = list(dict.fromkeys(devices))
    if jobs > len(unique) and not allow_sharing:
        raise LauncherRefused(
            f"JOBS={jobs} exceeds the {len(unique)} distinct device(s) {unique}; "
            "two trainers on one card is a memory decision to make on purpose -- "
            "set ALLOW_GPU_SHARING=1 if that is what you want")
    return [unique[index % len(unique)] for index in range(jobs)]


def _make_run(output_root, dataset, model, seed, phase, entry=None):
    entry = entry or {}
    config = ROOT / f"configs/paper/v4/{dataset}_{model}.yaml"
    digest = sha256_file(config)[:16]
    action = entry.get("execution_action", "train")
    # A reused run points at the directory that already holds its checkpoint, so
    # nothing writes into it and its provenance stays where it was produced.
    directory = (ROOT / entry["source_run_directory"]
                 if action == "reuse" and entry.get("source_run_directory")
                 else Path(output_root) / dataset / model / digest / f"seed{seed}")
    return Run(dataset=dataset, model=model, seed=int(seed), phase=int(phase),
               config=config, directory=directory,
               log=Path(output_root) / "logs" / f"{dataset}_{model}_seed{seed}.log",
               execution_action=action, entry=entry)


def plan_runs(output_root, plan):
    """The run list, in phase order, taken from the plan.

    A plan that enumerates its own runs is used verbatim; the two-dataset
    development plan predates that field and is derived from its datasets,
    models and seeds. Either way the enumeration comes from the plan, never
    from a constant in this file -- a launcher with `DATASETS = (...)` baked in
    silently ignores a dataset the plan added, which is exactly the class of
    error that makes a five-dataset experiment quietly a four-dataset one.
    """
    runs, seen = [], set()
    if plan.get("runs"):
        for entry in plan["runs"]:
            key = (entry["dataset"], entry["model"], int(entry["seed"]))
            if key in seen:
                raise LauncherRefused(f"duplicate run {key}")
            seen.add(key)
            runs.append(_make_run(output_root, *key, entry.get(
                "phase", 0 if int(entry["seed"]) == HEALTH_SEED else 1), entry))
    else:
        datasets = tuple(plan["datasets"])
        models = tuple(plan["models"])
        seeds = tuple(int(seed) for seed in plan["seeds"])
        health = seeds[0]
        for phase, phase_seeds in ((0, (health,)),
                                   (1, tuple(s for s in seeds if s != health))):
            for seed in phase_seeds:
                for dataset in datasets:
                    for model in models:
                        key = (dataset, model, seed)
                        if key in seen:
                            raise LauncherRefused(f"duplicate run {key}")
                        seen.add(key)
                        runs.append(_make_run(output_root, dataset, model, seed,
                                              phase))
    runs.sort(key=lambda run: (run.phase, run.seed, run.dataset, run.model))
    registered = {entry["path"] for entry in plan["configs"].values()}
    for run in runs:
        relative = str(run.config.relative_to(ROOT))
        if relative not in registered:
            raise LauncherRefused(f"{relative} is not in the registered plan")
    if len(runs) != plan["trial_budget"]:
        raise LauncherRefused(
            f"planned {len(runs)} runs, the plan registered "
            f"{plan['trial_budget']}")
    return runs


# --- run state ----------------------------------------------------------------


def run_state(run):
    """REUSED_COMPLETE | COMPLETE | FAILED | RESUMABLE | UNRESUMABLE | NEW."""
    directory = run.directory
    if run.execution_action == "reuse":
        # Its provenance was audited when the plan was frozen; the launcher
        # neither resumes it, restarts it, nor writes anything into it.
        return "REUSED_COMPLETE"
    if (directory / COMPLETE_MARKER).exists():
        return "COMPLETE"
    if (directory / FAILED_MARKER).exists():
        return "FAILED"
    if not run_artefacts(directory):
        return "NEW"
    return "RESUMABLE" if (directory / RUN_STATE_NAME).exists() else "UNRESUMABLE"


def current_epoch(run):
    """The last epoch line in the run's log, for the status view."""
    if not run.log.exists():
        return None
    lines = [line for line in run.log.read_text("utf-8", errors="replace").splitlines()
             if line.startswith("epoch ")]
    if not lines:
        return None
    try:
        return int(lines[-1].split()[1])
    except (IndexError, ValueError):
        return None


def build_command(run, output_root, plan_path=None, extra=()):
    """The trainer command, carrying the plan this launcher actually loaded.

    Hardcoding `--development_plan` here is what made the five-dataset launcher
    enumerate METR-LA correctly and then hand each run to the two-dataset plan,
    which rejected the config before CUDA. The plan path travels with the
    command, so the queue, the failure record and any resume all name the same
    plan.
    """
    command = [sys.executable, str(ROOT / "scripts/train_v4.py"),
               "--config", str(run.config.relative_to(ROOT)),
               "--device", run.device, "--seed", str(run.seed),
               "--output_root", str(output_root)]
    command += (["--plan", str(Path(plan_path))] if plan_path
                else ["--development_plan"])
    command.extend(extra)
    return command


def acquire_lock(run):
    """An exclusive advisory lock, so two launchers cannot share a run."""
    run.directory.mkdir(parents=True, exist_ok=True)
    handle = open(run.directory / LOCK_NAME, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.write(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    handle.flush()
    return handle


def release_lock(run):
    if run.lock is not None:
        try:
            fcntl.flock(run.lock.fileno(), fcntl.LOCK_UN)
            run.lock.close()
        except OSError:
            pass
        run.lock = None


def write_failure(run, returncode, command):
    tail = ""
    if run.log.exists():
        tail = "\n".join(
            run.log.read_text("utf-8", errors="replace").splitlines()[-40:])
    run.directory.mkdir(parents=True, exist_ok=True)
    (run.directory / FAILED_MARKER).write_text(json.dumps({
        "schema_version": 1,
        "run": run.name,
        "dataset": run.dataset, "model": run.model, "seed": run.seed,
        "exit_code": int(returncode),
        "command": " ".join(command),
        "device": run.device,
        "log": str(run.log),
        "log_tail": tail,
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "launcher_version": LAUNCHER_VERSION,
    }, indent=1, allow_nan=False) + "\n", "utf-8")


# --- the queue ----------------------------------------------------------------


class Launcher:

    def __init__(self, output_root, plan, slots, retry_failed=False,
                 restart_unresumable=False, plan_path=None):
        self.output_root = Path(output_root)
        self.plan = plan
        self.plan_path = plan_path
        self.slots = list(slots)
        self.retry_failed = retry_failed
        self.restart_unresumable = restart_unresumable
        self.runs = plan_runs(self.output_root, plan)
        self.active = []
        self.failed = []
        self.stopping = False
        (self.output_root / "logs").mkdir(parents=True, exist_ok=True)

    # -- signals --------------------------------------------------------------

    def install_signal_handlers(self):
        for name in (signal.SIGINT, signal.SIGTERM):
            signal.signal(name, self._on_signal)

    def _on_signal(self, signum, _frame):
        if self.stopping:
            return
        self.stopping = True
        self.say(f"received {signal.Signals(signum).name}; stopping {len(self.active)} "
                 "running job(s). last.pt is preserved; resume with the same command")
        for run in list(self.active):
            self.terminate(run)

    def terminate(self, run):
        """SIGTERM the trainer's whole process group, so no worker is orphaned."""
        if run.process is None or run.process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(run.process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            run.process.terminate()

    # -- reporting ------------------------------------------------------------

    def say(self, message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def status_rows(self):
        rows = []
        for run in self.runs:
            state = run_state(run)
            live = next((r for r in self.active if r is run), None)
            rows.append({
                "run": run.name, "phase": run.phase,
                "state": "RUNNING" if live else state,
                "device": run.device if live else "-",
                "pid": live.process.pid if live and live.process else "-",
                "epoch": current_epoch(run), "log": str(run.log),
            })
        return rows

    def print_status(self):
        rows = self.status_rows()
        counts = {}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        self.say("  ".join(f"{name}={value}" for name, value in sorted(counts.items())))
        for row in rows:
            if row["state"] in ("RUNNING", "FAILED"):
                self.say(f"   {row['state']:9s} {row['run']:26s} "
                         f"gpu={row['device']:8s} pid={row['pid']:<8} "
                         f"epoch={row['epoch']}")

    # -- execution ------------------------------------------------------------

    def start(self, run, device):
        if run.execution_action != "train":
            raise LauncherRefused(
                f"{run.name} is {run.execution_action}, not a training run; the "
                "launcher does not touch a reused checkpoint")
        state = run_state(run)
        extra = []
        if state == "RESUMABLE":
            extra = ["--resume"]
        elif state == "UNRESUMABLE":
            if not self.restart_unresumable:
                raise LauncherRefused(
                    f"{run.name}: {run.directory} is non-empty but has no "
                    f"{RUN_STATE_NAME}; pass --restart-unresumable to archive it "
                    "and start over rather than overwriting it")
            extra = ["--restart-incomplete"]
        elif state == "FAILED":
            if not self.retry_failed:
                raise LauncherRefused(
                    f"{run.name} is marked FAILED; pass --retry-failed to run it "
                    "again after reading the failure record")
            (run.directory / FAILED_MARKER).unlink()
            extra = (["--resume"] if (run.directory / RUN_STATE_NAME).exists()
                     else ["--restart-incomplete"])
        run.lock = acquire_lock(run)
        if run.lock is None:
            self.say(f"   {run.name} is locked by another launcher; skipping")
            return False
        run.device = device
        command = build_command(run, self.output_root, self.plan_path, extra)
        run.log.parent.mkdir(parents=True, exist_ok=True)
        handle = run.log.open("a", buffering=1)
        handle.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                     f"{' '.join(command)} ===\n")
        handle.flush()
        # Its own process group, so a signal reaches the trainer and its
        # DataLoader workers together and leaves nothing behind.
        run.process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        run._log_handle = handle
        run.started = time.time()
        self.active.append(run)
        self.say(f"START {run.name:26s} gpu={device} pid={run.process.pid} "
                 f"{'resume' if extra == ['--resume'] else 'fresh'}")
        return True

    def was_stopped_by_us(self, run):
        """Did *we* end this run, at a point it can be resumed from?

        Deliberately not "did it exit with a negative code". A trainer that
        segfaults or is OOM-killed also exits on a signal, and calling that a
        clean stop would hide a real failure behind the same message. The test
        is that the launcher was already stopping -- i.e. it forwarded the
        signal itself -- *and* the run left a `last.pt` to continue from.
        """
        return self.stopping and (run.directory / RUN_STATE_NAME).exists()

    def reap(self):
        for run in list(self.active):
            code = run.process.poll()
            if code is None:
                continue
            self.active.remove(run)
            release_lock(run)
            getattr(run, "_log_handle", None) and run._log_handle.close()
            minutes = (time.time() - run.started) / 60.0
            if code == 0 and (run.directory / COMPLETE_MARKER).exists():
                self.say(f"DONE  {run.name:26s} {minutes:.1f} min")
            elif self.was_stopped_by_us(run):
                # Not a failure: the user asked for this. No FAILED.json, so the
                # run stays RESUMABLE and the same command picks it up.
                epoch = current_epoch(run)
                self.say(f"STOPPED {run.name:24s} INTERRUPTED at epoch {epoch}, "
                         f"resumable after {minutes:.1f} min -- rerun the same "
                         "command to continue")
            else:
                write_failure(run, code, build_command(run, self.output_root,
                                                       self.plan_path))
                self.failed.append(run)
                self.say(f"FAIL  {run.name:26s} exit={code} after {minutes:.1f} min "
                         f"-- see {run.log}")

    def run_phase(self, phase):
        """One phase to completion. Returns False if anything failed."""
        pending = [run for run in self.runs if run.phase == phase
                   and run.execution_action == "train"
                   and run_state(run) != "COMPLETE"]
        skipped = [run for run in self.runs if run.phase == phase
                   and run.execution_action == "train"
                   and run_state(run) == "COMPLETE"]
        for run in skipped:
            self.say(f"SKIP  {run.name:26s} already COMPLETE")
        queue = list(pending)
        free = list(self.slots)
        while (queue or self.active) and not self.stopping:
            while queue and free and not self.failed and not self.stopping:
                device = free.pop(0)
                run = queue.pop(0)
                if not self.start(run, device):
                    free.append(device)
            time.sleep(POLL_SECONDS)
            before = len(self.active)
            self.reap()
            for _ in range(before - len(self.active)):
                free.append(self.slots[len(free) % len(self.slots)])
            if self.failed and self.active:
                self.say("a job in this phase failed; waiting for the rest to "
                         "finish, and not starting the next phase")
            if self.failed:
                queue = []
        return not self.failed and not self.stopping

    def execute(self):
        self.install_signal_handlers()
        self.say(f"launcher {LAUNCHER_VERSION}  plan {self.plan['plan_sha256'][:16]}  "
                 f"slots {self.slots}")
        for phase, label in ((0, "health (seed 32)"), (1, "remaining seeds")):
            self.say(f"--- phase {phase}: {label} ---")
            if not self.run_phase(phase):
                self.print_status()
                if self.stopping:
                    self.say("stopped by signal; no orphaned trainers")
                    return 130
                self.say(f"{len(self.failed)} run(s) failed in phase {phase}; "
                         "the remaining phases are not started")
                return 1
            self.print_status()
        self.say("all 12 runs COMPLETE")
        return 0


# --- entry point ---------------------------------------------------------------


def summarise(runs):
    reused = [run for run in runs if run.execution_action == "reuse"]
    training = [run for run in runs if run.execution_action == "train"]
    return {"total_result_slots": len(runs),
            "reused_complete": len(reused),
            "new_training": len(training),
            "gpu_jobs_to_execute": len(training),
            "reused": reused, "training": training}


def dry_run(output_root, plan, slots):
    runs = plan_runs(output_root, plan)
    counts = summarise(runs)
    print(f"plan     {plan['plan_sha256']}")
    print(f"output   {output_root}")
    print(f"slots    {slots}")
    print(f"\ntotal result slots: {counts['total_result_slots']}")
    print(f"reused complete:    {counts['reused_complete']}")
    print(f"new training:       {counts['new_training']}")
    print(f"GPU jobs to execute:{counts['gpu_jobs_to_execute']:>3d}")
    if counts["reused"]:
        print("\nreused (no GPU worker, never resumed, never written to):")
        for run in counts["reused"]:
            entry = run.entry
            print(f"  REUSED_COMPLETE {run.name:26s} "
                  f"source_plan={str(entry.get('source_plan_sha256',''))[:12]}... "
                  f"checkpoint_sha={str(entry.get('checkpoint_sha256',''))[:12]}...")
    for phase, label in ((0, "health (seed 32)"), (1, "remaining seeds")):
        members = [run for run in counts["training"] if run.phase == phase]
        print(f"\nphase {phase}: {label}  ({len(members)} runs to train, "
              f"at most {len(slots)} at a time)")
        for index, run in enumerate(members):
            print(f"  [{index + 1:2d}] {run.name:26s} gpu={slots[index % len(slots)]:8s} "
                  f"state={run_state(run):16s} {run.directory}")
    names = [run.name for run in runs]
    if len(names) != len(set(names)) or len(names) != plan["trial_budget"]:
        raise LauncherRefused(
            f"{len(names)} runs, {len(set(names))} unique, plan registered "
            f"{plan['trial_budget']}")
    declared = plan.get("execution")
    if declared and declared["gpu_jobs_to_execute"] != counts["gpu_jobs_to_execute"]:
        raise LauncherRefused(
            f"the plan declares {declared['gpu_jobs_to_execute']} GPU jobs, the "
            f"queue enumerated {counts['gpu_jobs_to_execute']}")
    print(f"\n{len(names)} unique result slots, no duplicates; "
          f"{counts['gpu_jobs_to_execute']} to train")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 1)))
    parser.add_argument("--devices", default=os.environ.get("DEVICES", "0"))
    parser.add_argument("--allow-gpu-sharing", action="store_true",
                        default=os.environ.get("ALLOW_GPU_SHARING") == "1")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--restart-unresumable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true",
                        help="print the queue state and exit")
    parser.add_argument("--plan", default=None,
                        help="a frozen plan JSON; defaults to the development plan")
    arguments = parser.parse_args(argv)

    plan, output_root = resolve_plan_and_root(arguments.plan)
    slots = assign_slots(arguments.jobs, parse_devices(arguments.devices),
                         arguments.allow_gpu_sharing)
    if arguments.dry_run:
        return dry_run(output_root, plan, slots)
    if arguments.status:
        launcher = Launcher(output_root, plan, slots, plan_path=arguments.plan)
        for row in launcher.status_rows():
            print(f"{row['state']:12s} phase{row['phase']} {row['run']:26s} "
                  f"epoch={row['epoch']}  {row['log']}")
        return 0
    return Launcher(output_root, plan, slots,
                    retry_failed=arguments.retry_failed,
                    restart_unresumable=arguments.restart_unresumable,
                    plan_path=arguments.plan).execute()


if __name__ == "__main__":
    raise SystemExit(main())

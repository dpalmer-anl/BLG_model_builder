#!/usr/bin/env python3
"""Run a subset of MCMC shell commands assigned to this Slurm task.

Each Slurm task (``SLURM_PROCID``) executes every *n*-th command from the
tasks file, where *n* = ``SLURM_NTASKS``.  Commands are run sequentially on
each rank.

Example (4 Slurm tasks, 10 commands)::

    rank 0: commands 0, 4, 8
    rank 1: commands 1, 5, 9
    rank 2: commands 2, 6
    rank 3: commands 3, 7

Usage
-----
On the cluster (inside an ``sbatch`` job with ``--ntasks=N``)::

    srun python run_mcmc_task_runner.py --tasks-file mcmc_task_lists/job_000.txt

Local smoke test (single process runs all commands)::

    python run_mcmc_task_runner.py --tasks-file mcmc_task_lists/job_000.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _slurm_rank_and_size(
    proc_id: int | None,
    ntasks: int | None,
) -> tuple[int, int]:
    if proc_id is not None and ntasks is not None:
        return int(proc_id), max(1, int(ntasks))
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
    size = int(os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
    return rank, max(1, size)


def load_tasks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        tasks = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    if not tasks:
        raise SystemExit(f"No tasks found in {path}")
    return tasks


def tasks_for_rank(tasks: list[str], rank: int, ntasks: int) -> list[str]:
    return [cmd for i, cmd in enumerate(tasks) if i % ntasks == rank]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCMC commands for one Slurm task rank.")
    parser.add_argument(
        "--tasks-file",
        required=True,
        help="Text file with one shell command per line.",
    )
    parser.add_argument(
        "--proc-id",
        type=int,
        default=None,
        help="Override rank (default: SLURM_PROCID or 0).",
    )
    parser.add_argument(
        "--ntasks",
        type=int,
        default=None,
        help="Override task count (default: SLURM_NTASKS or 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assigned commands without executing.",
    )
    args = parser.parse_args()

    rank, ntasks = _slurm_rank_and_size(args.proc_id, args.ntasks)
    all_tasks = load_tasks(args.tasks_file)
    mine = tasks_for_rank(all_tasks, rank, ntasks)

    print(
        f"[run_mcmc_task_runner] rank {rank}/{ntasks - 1}, "
        f"{len(mine)}/{len(all_tasks)} commands from {args.tasks_file}",
        flush=True,
    )
    for i, cmd in enumerate(mine, start=1):
        print(f"[run_mcmc_task_runner] rank {rank} cmd {i}/{len(mine)}: {cmd}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(
                f"[run_mcmc_task_runner] command failed (exit {result.returncode}): {cmd}",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(result.returncode)

    print(f"[run_mcmc_task_runner] rank {rank} finished.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Manage a local Virtuoso service using binaries from the active environment.

This is a COE-local replacement for KBQA-o1's Freebase wrapper script.
It provides start/stop/status commands and writes a tuned ``virtuoso.ini``
inside the specified DB directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def isql_port(http_port: int) -> int:
    return 10000 + http_port


def _resolve_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    print(
        f"Could not find '{name}' in PATH. Activate your conda env and install "
        "virtuoso-opensource (conda-forge).",
        file=sys.stderr,
    )
    sys.exit(1)


def _run_capture(command: str) -> str:
    res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or f"Command failed: {command}")
    return res.stdout.strip()


def _run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _compute_buffers(buffer_fraction: float) -> tuple[int, int, int]:
    mem_total_kb = int(_run_capture("cat /proc/meminfo | grep MemTotal | awk '{print $2}'"))
    number_of_buffers = int(mem_total_kb * buffer_fraction / 8)
    max_dirty_buffers = max(1, number_of_buffers // 2)
    return mem_total_kb, number_of_buffers, max_dirty_buffers


def _build_config(db_path: Path, http_port: int, number_of_buffers: int, max_dirty_buffers: int) -> str:
    return (
        "[Database]\n"
        f"DatabaseFile = {db_path}/virtuoso.db\n"
        f"ErrorLogFile = {db_path}/virtuoso.log\n"
        f"LockFile = {db_path}/virtuoso.lck\n"
        f"TransactionFile = {db_path}/virtuoso.trx\n"
        f"xa_persistent_file = {db_path}/virtuoso.pxa\n"
        "ErrorLogLevel = 7\n"
        "FileExtend = 200\n"
        "MaxCheckpointRemap = 2000\n"
        "Striping = 0\n"
        "TempStorage = TempDatabase\n"
        "\n"
        "[TempDatabase]\n"
        f"DatabaseFile = {db_path}/virtuoso-temp.db\n"
        f"TransactionFile = {db_path}/virtuoso-temp.trx\n"
        "MaxCheckpointRemap = 2000\n"
        "Striping = 0\n"
        "\n"
        "[Parameters]\n"
        f"ServerPort = {isql_port(http_port)}\n"
        "LiteMode = 0\n"
        "DisableUnixSocket = 1\n"
        "DisableTcpSocket = 0\n"
        "ServerThreads = 100\n"
        "CheckpointInterval = 60\n"
        "O_DIRECT = 1\n"
        "CaseMode = 2\n"
        "MaxStaticCursorRows = 100000\n"
        "CheckpointAuditTrail = 0\n"
        "AllowOSCalls = 0\n"
        "SchedulerInterval = 10\n"
        "DirsAllowed = .\n"
        "ThreadCleanupInterval = 0\n"
        "ThreadThreshold = 10\n"
        "ResourcesCleanupInterval = 0\n"
        "FreeTextBatchSize = 100000\n"
        "PrefixResultNames = 0\n"
        "RdfFreeTextRulesSize = 100\n"
        "IndexTreeMaps = 64\n"
        "MaxMemPoolSize = 200000000\n"
        f"NumberOfBuffers = {number_of_buffers}\n"
        f"MaxDirtyBuffers = {max_dirty_buffers}\n"
        "\n"
        "[SPARQL]\n"
        "ResultSetMaxRows = 50000\n"
        "MaxQueryCostEstimationTime = 600\n"
        "MaxQueryExecutionTime = 180\n"
        "\n"
        "[HTTPServer]\n"
        f"ServerPort = {http_port}\n"
        "Charset = UTF-8\n"
        "ServerThreads = 15\n"
    )


def start(db_path: Path, http_port: int, buffer_fraction: float) -> None:
    virtuoso_t = _resolve_binary("virtuoso-t")
    db_path.mkdir(parents=True, exist_ok=True)

    mem_total_kb, number_of_buffers, max_dirty_buffers = _compute_buffers(buffer_fraction)
    print(
        f"MemTotal={mem_total_kb} KB, NumberOfBuffers={number_of_buffers}, "
        f"MaxDirtyBuffers={max_dirty_buffers}"
    )

    config_path = db_path / "virtuoso.ini"
    config = _build_config(db_path, http_port, number_of_buffers, max_dirty_buffers)
    config_path.write_text(config)

    print(f"Starting Virtuoso on HTTP :{http_port} (ISQL :{isql_port(http_port)})")
    cmd = [virtuoso_t, "+configfile", str(config_path), "+wait"]
    proc = _run(cmd, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def _run_isql_statement(http_port: int, statement: str) -> None:
    isql = _resolve_binary("isql")
    cmd = [isql, f"localhost:{isql_port(http_port)}"]
    proc = subprocess.run(
        cmd,
        input=statement,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def stop(http_port: int) -> None:
    _run_isql_statement(http_port, "shutdown;\n")


def status(http_port: int) -> None:
    _run_isql_statement(http_port, "status();\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Virtuoso service for Freebase")
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("port", type=int, help="HTTP SPARQL port (e.g. 8890 or 3001)")
    parser.add_argument("-d", "--db-path", type=Path, help="Path to Virtuoso DB directory")
    parser.add_argument(
        "--buffer-fraction",
        type=float,
        default=0.15,
        help="Fraction of MemTotal used for NumberOfBuffers (default: 0.15)",
    )

    args = parser.parse_args()

    if args.action == "start":
        if args.db_path is None:
            parser.error("start requires -d/--db-path")
        if not args.db_path.exists():
            parser.error(f"DB path does not exist: {args.db_path}")
        start(args.db_path, args.port, args.buffer_fraction)
        return 0

    if args.action == "stop":
        stop(args.port)
        return 0

    status(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

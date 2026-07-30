#!/usr/bin/env python3
"""Prepare a complete, hash-recorded R1 input manifest for strict validation."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from workflow.common import read_json, write_json


WORKLOADS = ("fft", "matmul", "stencil", "stream")
REQUIRED_FILES = ("r1_metadata.json", "stats.txt", "status.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(r1_root: Path, output: Path, l1d_size: str = "32kB",
            l2_size: str = "512kB") -> dict:
    """Validate all four fixed R1 points before atomically writing a manifest."""
    if l1d_size != "32kB" or l2_size != "512kB":
        raise ValueError(
            "strict R1 validation requires exactly l1d_size=32kB and l2_size=512kB"
        )
    r1_root = r1_root.resolve()
    records = []
    for workload in WORKLOADS:
        point = r1_root / workload / f"l1d_{l1d_size}" / f"l2_{l2_size}"
        missing = [name for name in REQUIRED_FILES if not (point / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{workload} R1 point is incomplete at {point}: {', '.join(missing)}"
            )
        status = read_json(point / "status.json")
        if status.get("state") != "success":
            raise ValueError(f"{workload} R1 status is not success: {point / 'status.json'}")
        metadata = read_json(point / "r1_metadata.json")
        expected = {"workload": workload, "l1d_size": l1d_size, "l2_size": l2_size}
        mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatched:
            raise ValueError(
                f"{workload} R1 metadata does not match requested point: {', '.join(mismatched)}"
            )
        instruction_window_scope = metadata.get("instruction_window_scope")
        if not isinstance(instruction_window_scope, str) or not instruction_window_scope.strip():
            raise ValueError(f"{workload} R1 metadata has no instruction_window_scope")
        files = {
            name: {"path": str((point / name).resolve()), "sha256": sha256(point / name)}
            for name in REQUIRED_FILES
        }
        records.append({
            "workload": workload,
            "l1d_size": l1d_size,
            "l2_size": l2_size,
            "directory": str(point.resolve()),
            "instruction_window_scope": instruction_window_scope,
            "files": files,
        })
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "r1_root": str(r1_root),
        "workloads": list(WORKLOADS),
        "l1d_size": l1d_size,
        "l2_size": l2_size,
        "points": records,
    }
    write_json(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l1d-size", default="32kB")
    parser.add_argument("--l2-size", default="512kB")
    args = parser.parse_args()
    result = prepare(args.r1_root, args.output, args.l1d_size, args.l2_size)
    print(f"prepared {len(result['points'])} strict R1 input points")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Patch, build, and install the five semantic-ROI workload binaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = PROJECT_ROOT / "benchmarks/semantic_roi"
PATCH_DIR = ROI_DIR / "patches"
DEFAULT_GEM5_ROOT = PROJECT_ROOT / "tools/src/gem5"


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def apply_patch(source_root: Path, patch: Path,
                markers: tuple[tuple[Path, str], ...]) -> None:
    states = [
        needle in marker.read_text(encoding="utf-8")
        for marker, needle in markers
    ]
    if all(states):
        return
    if any(states):
        incomplete = [str(marker) for (marker, _), found in zip(markers, states)
                      if not found]
        raise RuntimeError(
            f"semantic patch is only partially applied ({patch}); missing "
            + ", ".join(incomplete)
        )
    run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=source_root)
    for marker, needle in markers:
        if needle not in marker.read_text(encoding="utf-8"):
            raise RuntimeError(
                f"patch did not materialize expected marker in {marker}"
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-benchmark-root", type=Path,
        default=PROJECT_ROOT / "benchmarks",
        help="benchmark tree containing downloaded SPLASH-2 and STREAM sources",
    )
    parser.add_argument("--gem5-root", type=Path, default=DEFAULT_GEM5_ROOT)
    args = parser.parse_args()
    external = args.external_benchmark_root.resolve()
    gem5_root = args.gem5_root.resolve()
    if not (gem5_root / "include/gem5/m5ops.h").is_file():
        raise FileNotFoundError(f"gem5 m5ops headers not found under {gem5_root}")

    patch_specs = (
        ("fft-semantic-roi.patch", (
            (external / "src/splash2/kernels/fft/src/fft.C", "clip_roi_boundary"),
            (external / "src/splash2/kernels/fft/src/makefile", "ROI_OBJECTS"),
        )),
        ("cholesky-semantic-roi.patch", (
            (external / "src/splash2/kernels/cholesky/src/solve.C", "clip_roi_boundary"),
            (external / "src/splash2/kernels/cholesky/src/makefile", "ROI_OBJECTS"),
        )),
        ("stream-semantic-roi.patch", (
            (external / "src/stream/stream.c", "clip_roi_boundary"),
            (external / "src/stream/Makefile", "ROI_OBJECTS"),
        )),
    )
    for patch_name, markers in patch_specs:
        for marker, _ in markers:
            if not marker.is_file():
                raise FileNotFoundError(
                    f"external benchmark source is missing: {marker}"
                )
        apply_patch(external, PATCH_DIR / patch_name, markers)

    run(["make", "GEM5_ROOT=" + str(gem5_root)], cwd=ROI_DIR)
    roi_arg = "CLIP_ROI_DIR=" + str(ROI_DIR)
    run(["make", "M4=m4", roi_arg, "fft"],
        cwd=external / "src/splash2/kernels/fft/src")
    run(["make", "M4=m4", roi_arg, "cholesky"],
        cwd=external / "src/splash2/kernels/cholesky/src")
    run(["make", roi_arg, "stream_c.exe"], cwd=external / "src/stream")
    run(["make", "GEM5_ROOT=" + str(gem5_root)],
        cwd=PROJECT_ROOT / "benchmarks/src/matmul")
    run(["make", "GEM5_ROOT=" + str(gem5_root)],
        cwd=PROJECT_ROOT / "benchmarks/src/stencil")

    binaries = {
        "fft": external / "src/splash2/kernels/fft/src/fft",
        "cholesky": external / "src/splash2/kernels/cholesky/src/cholesky",
        "stream": external / "src/stream/stream_c.exe",
        "matmul": PROJECT_ROOT / "benchmarks/src/matmul/matmul",
        "stencil": PROJECT_ROOT / "benchmarks/src/stencil/stencil",
    }
    destination = PROJECT_ROOT / "benchmarks/bin"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "binaries": {}, "patches": {}}
    for name, source in binaries.items():
        target = destination / name
        shutil.copy2(source, target)
        manifest["binaries"][name] = {
            "path": str(target.resolve()), "sha256": sha256(target),
        }
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        manifest["patches"][patch.name] = sha256(patch)
    manifest["shared_source_sha256"] = {
        name: sha256(ROI_DIR / name) for name in ("clip_roi.c", "clip_roi.h")
    }
    (destination / "semantic_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    cholesky_input = external / "inputs/cholesky/tk14.O"
    if cholesky_input.is_file():
        input_target = PROJECT_ROOT / "benchmarks/inputs/cholesky/tk14.O"
        input_target.parent.mkdir(parents=True, exist_ok=True)
        if input_target.resolve() != cholesky_input.resolve():
            shutil.copy2(cholesky_input, input_target)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Shared integrity checks for reusable transient-ROM evidence."""

from __future__ import annotations

from pathlib import Path

from workflow.common import sha256_file


ROM_CLASSIFICATION: dict[str, object] = {
    "thermal_mode": "transient-rom",
    "non_formal": True,
    "paper_equivalent": False,
}


def require_rom_classification(value: object, context: str) -> dict:
    """Return a classified record or reject one from another thermal mode."""
    if not isinstance(value, dict) or any(
        value.get(field) != expected
        for field, expected in ROM_CLASSIFICATION.items()
    ):
        raise ValueError(f"{context} classification is invalid")
    return value


def require_regular_descendant(root: Path, relative: str, context: str) -> Path:
    """Return a regular package file without following symlinked components."""
    root = Path(root)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{context} must be package-relative")
    if root.is_symlink():
        raise ValueError(f"{context} contains a symlink")
    current = root
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{context} contains a symlink")
    if not current.is_file() or current.is_symlink():
        raise ValueError(f"{context} must be a regular file")
    return current


def require_new_output_root(path: Path, forbidden_roots: list[Path]) -> Path:
    """Require an output root disjoint from every package or read-only input."""
    output = Path(path).resolve(strict=False)
    for forbidden in forbidden_roots:
        root = Path(forbidden).resolve(strict=False)
        if output == root or output in root.parents or root in output.parents:
            raise ValueError("output root overlaps a forbidden input or package root")
    return output


def sha256_identity(path: Path) -> str:
    """Return the canonical SHA-256 identity used in ROM provenance records."""
    return "sha256:" + sha256_file(path)

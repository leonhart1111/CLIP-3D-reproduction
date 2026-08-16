#!/usr/bin/env bash
# Apply the audited McPAT metric patch and rebuild the executable from a clean target.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

mcpat_source_dir=${MCPAT_SOURCE_DIR:-tools/src/mcpat}
mcpat_patch_file=${MCPAT_PATCH_FILE:-patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch}
provenance_file=${MCPAT_PROVENANCE_FILE:-tools/build/mcpat/build_provenance.json}
mcpat_cxx=${MCPAT_CXX:-g++}
mcpat_cc=${MCPAT_CC:-gcc}
binary="$mcpat_source_dir/mcpat"

[[ -d "$mcpat_source_dir" ]] || {
    echo "McPAT source directory does not exist: $mcpat_source_dir" >&2
    exit 1
}
[[ -f "$mcpat_patch_file" ]] || {
    echo "McPAT patch does not exist: $mcpat_patch_file" >&2
    exit 1
}

patch_state=
if patch -d "$mcpat_source_dir" -p1 --dry-run -R < "$mcpat_patch_file"; then
    patch_state=already_applied
elif patch -d "$mcpat_source_dir" -p1 --dry-run < "$mcpat_patch_file"; then
    patch -d "$mcpat_source_dir" -p1 < "$mcpat_patch_file"
    patch_state=applied
else
    echo "McPAT source is partially patched or incompatible; aborting." >&2
    exit 1
fi

build_started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
make -C "$mcpat_source_dir" clean CXX="$mcpat_cxx" CC="$mcpat_cc"
make -C "$mcpat_source_dir" CXX="$mcpat_cxx" CC="$mcpat_cc"
build_completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

[[ -x "$binary" ]] || {
    echo "McPAT build did not produce an executable: $binary" >&2
    exit 1
}
marker_strings=$(strings "$binary")
grep -Fq CLIP_MCPAT_CACTI_P_V1 <<< "$marker_strings"

mkdir -p "$(dirname "$provenance_file")"
python3 - "$mcpat_patch_file" "$binary" "$provenance_file" "$patch_state" \
    "$build_started_at_utc" "$build_completed_at_utc" "$mcpat_cxx" "$mcpat_cc" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

patch = Path(sys.argv[1])
binary = Path(sys.argv[2])
destination = Path(sys.argv[3])
patch_state = sys.argv[4]
started = sys.argv[5]
completed = sys.argv[6]
cxx = sys.argv[7]
cc = sys.argv[8]

payload = {
    "schema_version": 1,
    "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "patch_state": patch_state,
    "build_commands": [
        "make clean CXX={0} CC={1}".format(cxx, cc),
        "make CXX={0} CC={1}".format(cxx, cc),
    ],
    "build_started_at_utc": started,
    "build_completed_at_utc": completed,
    "built_at_utc": completed,
}
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

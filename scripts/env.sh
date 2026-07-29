#!/usr/bin/env bash

# Source this file instead of executing it:
#   source /home/zyjiang/Agenticflow/CLIP/scripts/env.sh

export CLIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GEM5_ROOT="$CLIP_ROOT/tools/src/gem5"
export MCPAT_ROOT="$CLIP_ROOT/tools/src/mcpat"
export CACTI_ROOT="$CLIP_ROOT/tools/src/cacti"
export HOTSPOT_ROOT="$CLIP_ROOT/tools/src/hotspot"

export CLIP_CONFIGS="$CLIP_ROOT/configs"
export CLIP_BENCHMARKS="$CLIP_ROOT/benchmarks"
export CLIP_RUNS="$CLIP_ROOT/runs"
export CLIP_RESULTS="$CLIP_ROOT/results"

# The EDA nodes use an older system glibc.  A login environment managed by
# Nix may put a newer GCC runtime in LD_LIBRARY_PATH; that libstdc++ then asks
# the system loader for GLIBC_2.32--2.38 and prevents the locally built McPAT
# (and potentially CACTI/gem5) from starting.  Prefer the node's compatible
# system/GCC-Toolset runtime.  Keeping the previous path at the end preserves
# unrelated user libraries while resolving libstdc++/libgcc compatibly first.
CLIP_GCC_TOOLSET_ROOT=/opt/rh/gcc-toolset-13/root/usr
CLIP_HOST_LIBRARY_PATH=/lib64:/usr/lib64
if [[ -d "$CLIP_GCC_TOOLSET_ROOT/lib64" ]]; then
    CLIP_HOST_LIBRARY_PATH="$CLIP_GCC_TOOLSET_ROOT/lib64:$CLIP_GCC_TOOLSET_ROOT/lib:$CLIP_HOST_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$CLIP_HOST_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -x "$CLIP_ROOT/.venv/bin/python" ]]; then
    export VIRTUAL_ENV="$CLIP_ROOT/.venv"
    export PATH="$CLIP_ROOT/.venv/bin:$CLIP_ROOT/tools/install/bin:$PATH"
else
    export PATH="$CLIP_ROOT/tools/install/bin:$PATH"
fi

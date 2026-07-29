#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="$ROOT/tools/src"
MANIFEST="$ROOT/manifests/tool_versions.tsv"
WRITE_MANIFEST=0

if [[ "${1:-}" == "--write-manifest" ]]; then
    WRITE_MANIFEST=1
fi

binary_for() {
    case "$1" in
        gem5) printf '%s\n' "$SRC_ROOT/gem5/build/X86/gem5.opt" ;;
        mcpat) printf '%s\n' "$SRC_ROOT/mcpat/mcpat" ;;
        cacti) printf '%s\n' "$SRC_ROOT/cacti/cacti" ;;
        hotspot) printf '%s\n' "$SRC_ROOT/hotspot/hotspot" ;;
    esac
}

status=0
for tool in gem5 mcpat cacti hotspot; do
    directory="$SRC_ROOT/$tool"
    binary="$(binary_for "$tool")"
    if [[ ! -d "$directory" || ! -x "$binary" ]]; then
        printf '[missing] %-8s source=%s binary=%s\n' "$tool" "$directory" "$binary"
        status=1
        continue
    fi
    if [[ -d "$directory/.git" ]]; then
        version="$(git -C "$directory" describe --tags --always --dirty)"
        commit="$(git -C "$directory" rev-parse --short=12 HEAD)"
    else
        version="archive-extract"
        commit="sha256-in-manifests/source_archives.sha256"
    fi
    printf '[ok]      %-8s %-18s %-44s %s\n' "$tool" "$version" "$commit" "$binary"
done

if [[ "$WRITE_MANIFEST" -eq 1 ]]; then
    if [[ "$status" -ne 0 ]]; then
        printf 'Refusing to write manifest while a tool is missing.\n' >&2
        exit "$status"
    fi
    {
        printf 'tool\tdescribe\tcommit_or_archive\tbinary\n'
        for tool in gem5 mcpat cacti hotspot; do
            directory="$SRC_ROOT/$tool"
            binary="$(binary_for "$tool")"
            if [[ -d "$directory/.git" ]]; then
                printf '%s\t%s\t%s\t%s\n' "$tool" \
                    "$(git -C "$directory" describe --tags --always --dirty)" \
                    "$(git -C "$directory" rev-parse HEAD)" "$binary"
            else
                printf '%s\tarchive-extract\t%s\t%s\n' "$tool" \
                    'see manifests/source_archives.sha256' "$binary"
            fi
        done
    } > "$MANIFEST"
    printf 'Updated %s\n' "$MANIFEST"
fi

exit "$status"

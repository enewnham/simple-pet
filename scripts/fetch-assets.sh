#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
source_dir=""
output_dir=""
pet="builtin"
ref=""

usage() {
    cat <<'EOF'
Usage: scripts/fetch-assets.sh [--pet builtin|NAME] [--source PATH] [--output PATH] [--ref COMMIT]

Fetch OpenPets and extract PNG frames with native macOS image libraries.
The default pet is the bundled Hoodie Cat. --source uses an existing checkout.
--pet NAME looks in assets/pets/NAME, apps/desktop/assets/pets/NAME, or pets/NAME.
A source pet package can also be supplied directly with --source.
The default output is assets/pets/NAME. Generated clips and metadata are replaced.
Each source must include LICENSE. Custom packs can declare clips in manifest.json.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pet|--source|--output|--ref)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            case "$1" in
                --pet) pet="$2" ;;
                --source) source_dir="$2" ;;
                --output) output_dir="$2" ;;
                --ref) ref="$2" ;;
            esac
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$output_dir" ]] || output_dir="$repo_root/assets/pets/$pet"

[[ "$(uname -s)" == "Darwin" ]] || { echo "Asset extraction requires macOS." >&2; exit 1; }
[[ -n "$output_dir" && -n "$pet" ]] || { echo "Output and pet must be nonempty." >&2; exit 2; }
if [[ -n "$source_dir" && -n "$ref" ]]; then
    echo "--ref applies to downloaded checkouts. Select the ref in your --source checkout." >&2
    exit 2
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/pet-assets.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
if [[ -z "$source_dir" ]]; then
    source_dir="$work_dir/openpets"
    git clone --quiet --depth 1 https://github.com/OpenPetsHQ/openpets.git "$source_dir"
    if [[ -n "$ref" ]]; then
        git -C "$source_dir" fetch --quiet --depth 1 origin "$ref"
        git -C "$source_dir" checkout --quiet --detach FETCH_HEAD
    fi
fi
[[ -d "$source_dir" ]] || { echo "Source directory does not exist: $source_dir" >&2; exit 1; }
source_dir="$(cd "$source_dir" && pwd)"
commit="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || printf 'unknown')"

swiftc -O -module-cache-path "$work_dir/module-cache" "$repo_root/scripts/import-assets.swift" -o "$work_dir/import-assets"
"$work_dir/import-assets" "$source_dir" "$output_dir" "$pet" "$commit"

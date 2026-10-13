#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
source_dir=""
zip_file=""
source_origin=""
output_dir=""
pet=""
ref=""

usage() {
    cat <<'EOF'
Usage: scripts/fetch-assets.sh [--pet NAME] [--source DIRECTORY | --zip FILE] [--output PATH] [--ref COMMIT]

Fetch OpenPets and extract PNG frames with native macOS image libraries.
--source uses an existing checkout or pet package. Select it with --pet NAME.
--zip imports a local downloaded pet archive. Its pet ID supplies the default name.
--pet NAME looks in assets/pets/NAME, apps/desktop/assets/pets/NAME, or pets/NAME.
A source pet package can also be supplied directly with --source.
The default output is assets/pets/NAME. Generated clips and metadata are replaced.
Custom packs can declare clips in manifest.json.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pet|--source|--zip|--output|--ref)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            case "$1" in
                --pet) pet="$2" ;;
                --source) source_dir="$2" ;;
                --zip) zip_file="$2" ;;
                --output) output_dir="$2" ;;
                --ref) ref="$2" ;;
            esac
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$(uname -s)" == "Darwin" ]] || { echo "Asset extraction requires macOS." >&2; exit 1; }
if [[ -n "$source_dir" && -n "$zip_file" ]]; then
    echo "Choose --source or --zip." >&2
    exit 2
fi
if [[ -n "$ref" && ( -n "$source_dir" || -n "$zip_file" ) ]]; then
    echo "--ref applies to downloaded checkouts." >&2
    exit 2
fi

if [[ -z "$zip_file" && -z "$pet" ]]; then
    echo "Select a pet with --pet NAME or --zip FILE." >&2
    exit 2
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/pet-assets.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
if [[ -n "$zip_file" ]]; then
    python3 "$repo_root/scripts/unpack-assets.py" "$zip_file" "$work_dir/zip" > "$work_dir/source.json"
    source_dir="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["directory"])' "$work_dir/source.json")"
    source_origin="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["origin"])' "$work_dir/source.json")"
    [[ -n "$pet" ]] || pet="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pet"])' "$work_dir/source.json")"
elif [[ -z "$source_dir" ]]; then
    source_dir="$work_dir/openpets"
    git clone --quiet --depth 1 https://github.com/OpenPetsHQ/openpets.git "$source_dir"
    if [[ -n "$ref" ]]; then
        git -C "$source_dir" fetch --quiet --depth 1 origin "$ref"
        git -C "$source_dir" checkout --quiet --detach FETCH_HEAD
    fi
fi
[[ -n "$output_dir" ]] || output_dir="$repo_root/assets/pets/$pet"
[[ -d "$source_dir" ]] || { echo "Source directory does not exist: $source_dir" >&2; exit 1; }
source_dir="$(cd "$source_dir" && pwd)"
commit="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || printf 'unknown')"

swiftc -O -module-cache-path "$work_dir/module-cache" "$repo_root/scripts/import-assets.swift" -o "$work_dir/import-assets"
"$work_dir/import-assets" "$source_dir" "$output_dir" "$pet" "$commit" "$source_origin"

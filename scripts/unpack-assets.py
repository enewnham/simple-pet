#!/usr/bin/env python3
"""Extract a local pet ZIP into a temporary source directory."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


MAX_UNPACKED_BYTES = 128 * 1024 * 1024


def is_pack(directory):
    files = ("pet.json", "manifest.json", "spritesheet.webp", "spritesheet.png", "idle.png", "rest.png")
    return any((directory / name).is_file() for name in files) or any((directory / name).is_dir() for name in ("idle", "rest"))


def unpack(archive, destination):
    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = []
        names = set()
        total = 0
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or any(ord(char) < 32 for char in entry.filename):
                raise ValueError("ZIP entries must stay inside the extracted directory")
            mode = entry.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError("ZIP entries must be regular files or directories")
            if "__MACOSX" in path.parts or path.name == ".DS_Store":
                continue
            name = str(path).casefold()
            if name in names:
                raise ValueError("ZIP contains duplicate paths")
            names.add(name)
            total += entry.file_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("Pet ZIP exceeds 128 MiB uncompressed")
            target = destination.joinpath(*path.parts)
            entries.append((entry, target))
        destination.mkdir(parents=True, exist_ok=True)
        for entry, target in entries:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    if is_pack(destination):
        source = destination
    else:
        packs = [item for item in destination.iterdir() if item.is_dir() and is_pack(item)]
        if len(packs) != 1:
            raise ValueError("ZIP must contain one pet at its root or in one top-level folder")
        source = packs[0]
    identifier = None
    for filename, key in (("pet.json", "id"), ("manifest.json", "pet")):
        metadata = source / filename
        if metadata.is_file():
            data = json.loads(metadata.read_text())
            if not isinstance(data, dict):
                raise ValueError(f"{filename} must contain a JSON object")
            identifier = data.get(key)
            if identifier:
                break
    identifier = identifier or archive.stem
    if not isinstance(identifier, str) or not identifier or identifier in (".", "..") or "/" in identifier or "\\" in identifier:
        raise ValueError("Pet ID must be a directory name without slashes")
    return {"directory": str(source), "pet": identifier, "origin": str(archive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(unpack(args.archive, args.destination)))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"ZIP import failed: {error}\n")


if __name__ == "__main__":
    main()

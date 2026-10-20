import json
from pathlib import Path
import runpy
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
UNPACK = runpy.run_path(str(ROOT / "scripts" / "unpack-assets.py"))["unpack"]


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
    + png_chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00\xff" * 2) * 2))
    + png_chunk(b"IEND", b"")
)


class ZipAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pet-zip-")
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / "my-pet.zip"
        self.extracted = self.directory / "extracted"

    def tearDown(self):
        self.temporary.cleanup()

    def write_zip(self, entries):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in entries:
                archive.writestr(name, data)

    def test_root_pack_uses_metadata_name(self):
        self.write_zip([("pet.json", '{"id":"little-cat"}'), ("idle.png", PNG)])
        result = UNPACK(self.archive, self.extracted)
        self.assertEqual(
            result,
            {
                "directory": str(self.extracted.resolve()),
                "pet": "little-cat",
                "origin": str(self.archive.resolve()),
            },
        )
        self.assertEqual((self.extracted / "idle.png").read_bytes(), PNG)

    def test_wrapped_pack_ignores_finder_files_and_uses_filename(self):
        self.write_zip(
            [
                ("download/idle.png", PNG),
                ("__MACOSX/._idle.png", "ignored"),
                ("download/.DS_Store", "ignored"),
            ]
        )
        result = UNPACK(self.archive, self.extracted)
        self.assertEqual(
            result["directory"], str((self.extracted / "download").resolve())
        )
        self.assertEqual(result["pet"], "my-pet")
        self.assertFalse((self.extracted / "__MACOSX").exists())
        self.assertFalse((self.extracted / "download" / ".DS_Store").exists())

    def test_manifest_supplies_name(self):
        self.write_zip([("manifest.json", '{"pet":"quiet-dog"}')])
        self.assertEqual(UNPACK(self.archive, self.extracted)["pet"], "quiet-dog")

    def test_unsafe_paths_are_rejected_before_extraction(self):
        for name in ("../outside.png", "/outside.png", "folder\\outside.png"):
            with self.subTest(name=name):
                self.write_zip([("idle.png", PNG), (name, PNG)])
                with self.assertRaises(ValueError):
                    UNPACK(self.archive, self.extracted)
                self.assertFalse(self.extracted.exists())

    def test_symlinks_and_case_collisions_are_rejected(self):
        link = zipfile.ZipInfo("idle.png")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for entries in ([(link, "../outside")], [("idle.png", PNG), ("IDLE.PNG", PNG)]):
            self.write_zip(entries)
            with self.assertRaises(ValueError):
                UNPACK(self.archive, self.extracted)
            self.assertFalse(self.extracted.exists())

    def test_multiple_pets_are_rejected(self):
        self.write_zip([("one/idle.png", PNG), ("two/idle.png", PNG)])
        with self.assertRaisesRegex(ValueError, "one pet"):
            UNPACK(self.archive, self.extracted)

    def test_corrupt_zip_reports_an_error(self):
        self.archive.write_bytes(b"not a zip")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "unpack-assets.py"),
                str(self.archive),
                str(self.extracted),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("ZIP import failed", result.stderr)

    @unittest.skipUnless(sys.platform == "darwin", "Asset preparation requires macOS")
    def test_fetch_script_imports_and_preserves_installed_pack_on_failure(self):
        self.write_zip([("pet.json", '{"id":"little-cat"}'), ("idle.png", PNG)])
        output = self.directory / "installed"
        command = [
            str(ROOT / "scripts" / "fetch-assets.sh"),
            "--zip",
            str(self.archive),
            "--output",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["pet"], "little-cat")
        self.assertEqual(manifest["source"], str(self.archive.resolve()))
        frame = output / manifest["clips"][manifest["rest"]]["frames"][0]
        self.assertTrue(frame.is_file())
        before = manifest_path.read_bytes()
        self.write_zip([("idle.png", b"broken image")])
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(manifest_path.read_bytes(), before)
        self.assertTrue(frame.is_file())

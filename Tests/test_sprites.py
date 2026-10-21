import json
import tempfile
import unittest
from pathlib import Path

from pet_daemon.behavior.sprites import Clip, Pack
from pet_daemon.runtime import Point


class SpriteTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="pet-sprites-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.directory / "rest.png").touch()

    def load(self, **clip: object) -> Pack:
        manifest = {
            "pet": "cat",
            "rest": "rest",
            "clips": {
                "rest": {"frames": ["rest.png"], "fps": 2, "role": "rest", **clip}
            },
        }
        (self.directory / "manifest.json").write_text(json.dumps(manifest))
        return Pack(self.directory)

    def test_manifest_exposes_validated_clips_and_geometry(self) -> None:
        pack = self.load(max_repeats=3)
        self.assertEqual(pack.pet, "cat")
        self.assertEqual(
            pack.clips[pack.rest], Clip("rest", ("rest.png",), 2, "rest", 3)
        )
        self.assertEqual(pack.gaze_anchor, Point(0.5, 0.5))
        self.assertEqual(pack.gaze, ())

    def test_invalid_clip_fields_are_rejected_at_load(self) -> None:
        for fields in (
            {"fps": True},
            {"fps": 0},
            {"fps": "2"},
            {"frames": [2]},
            {"frames": ["../outside.png"]},
            {"frames": []},
            {"max_repeats": True},
            {"max_repeats": 0},
            {"role": None},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.load(**fields)

    def test_invalid_manifest_shapes_are_rejected_at_load(self) -> None:
        for manifest in (
            [],
            {"pet": "cat", "rest": "rest", "clips": []},
            {"pet": "cat", "rest": "rest", "clips": {"rest": None}},
        ):
            with (
                self.subTest(manifest=manifest),
                self.assertRaises((TypeError, ValueError)),
            ):
                (self.directory / "manifest.json").write_text(json.dumps(manifest))
                Pack(self.directory)

import base64
import json
import math
from pathlib import Path
import tempfile
import unittest

from pet_daemon.behavior import Behavior
from pet_daemon.config import BehaviorConfig
from pet_daemon.sessions import Sessions
from pet_daemon.sprites import Pack


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=')


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RandomChoices:
    def __init__(self):
        self.angle = 0
        self.walk = False
        self.repeat_calls = []

    def random(self):
        return 0 if self.walk else 1

    def uniform(self, low, high):
        return self.angle if (low, high) == (0, 2 * math.pi) else low

    def randint(self, low, high):
        self.repeat_calls.append((low, high))
        return high

    def choice(self, choices):
        return choices[0]


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pet-behavior-')
        self.addCleanup(self.temp.cleanup)
        self.clock = Clock()
        self.rng = RandomChoices()
        self.output = []
        self.sessions = Sessions(self.clock)

    def make_behavior(self, clips=None, gaze=False, **config):
        directory = Path(self.temp.name)
        (directory / 'pixel.png').write_bytes(PNG)
        definitions = {'quiet-pose': {'role': 'rest', 'frames': ['pixel.png'], 'fps': 1}}
        for name, values in (clips or {}).items():
            definitions[name] = {'frames': ['pixel.png'] * values.pop('frames', 1), 'fps': values.pop('fps', 1), **values}
        manifest = {'pet': 'different-character', 'rest': 'quiet-pose', 'clips': definitions}
        if gaze:
            manifest['gaze'] = [f'look-{index}' for index in range(16)]
            definitions.update({name: {'frames': ['pixel.png'], 'fps': 1, 'role': 'gaze'} for name in manifest['gaze']})
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        defaults = {'idle_pause': (100, 100), 'working_pause': (100, 100), 'work_burst_seconds': (2, 2), 'walk_seconds': (2, 2)}
        behavior = Behavior(Pack(directory), BehaviorConfig(**(defaults | config)), self.clock, self.rng, self.output.append)
        self.assertTrue(behavior.renderer_status(self.status()))
        return behavior

    def status(self, **values):
        return {'kind': 'renderer', 'event': 'status', 'x': 100, 'y': 100, 'width': 200, 'height': 160,
                'screen': {'x': 0, 'y': 0, 'width': 600, 'height': 400},
                'pointer': {'x': 2000, 'y': 2000, 'moving': False},
                'dragging': False, 'menu': False, 'speech': False} | values

    def start(self):
        self.sessions.handle({'harness': 'cli', 'event': 'intent', 'session': 'one', 'goal': 'Build project'})
        self.sessions.handle({'harness': 'cli', 'event': 'tool_start', 'session': 'one', 'call': 'build', 'tool': 'build'})

    def finish(self):
        self.sessions.handle({'harness': 'cli', 'event': 'turn_done', 'session': 'one'})

    def animations(self):
        return [packet['animation'] for packet in self.output if 'animation' in packet]

    def clean_completion(self):
        self.start()
        self.finish()
        self.assertTrue(self.sessions.clean_convergence)

    def test_gesture_repeats_whole_cycles_with_random_range_and_pack_limit(self):
        behavior = self.make_behavior({'little-hop': {'role': 'celebrate', 'frames': 5, 'fps': 5, 'max_repeats': 2}})
        behavior.tick('idle', self.sessions)
        self.clean_completion()
        self.assertTrue(behavior.gesture('little-hop', self.sessions))
        animation = self.animations()[-1]
        self.assertEqual(animation['id'], 'little-hop')
        self.assertEqual(animation['repeats'], 2)
        self.assertEqual(self.rng.repeat_calls, [(1, 2)])
        self.assertEqual(behavior.pending['until'], self.clock() + 3)
        self.assertFalse(behavior.gesture('little-hop', self.sessions))

    def test_repeat_budget_keeps_long_clips_complete_and_plays_them_once(self):
        behavior = self.make_behavior({'slow-bounce': {'role': 'celebrate', 'frames': 10, 'fps': 2},
                                       'quick-bounce': {'role': 'celebrate', 'frames': 3, 'fps': 2}}, repeat_range=(2, 9), repeat_budget=4)
        behavior.tick('idle', self.sessions)
        self.clean_completion()
        self.assertTrue(behavior.gesture('slow-bounce', self.sessions))
        self.assertEqual(self.animations()[-1]['repeats'], 1)
        behavior.completion({'event': 'animation', 'command_id': self.animations()[-1]['command_id'], 'result': 'finished'})
        self.clock.advance(30)
        self.assertTrue(behavior.gesture('quick-bounce', self.sessions))
        self.assertEqual(self.animations()[-1]['repeats'], 2)

    def test_rest_only_pack_stays_still_and_rejects_unavailable_gestures(self):
        behavior = self.make_behavior()
        self.rng.walk = True
        for _ in range(4):
            behavior.tick('idle', self.sessions)
            self.clock.advance(101)
        self.assertFalse(any('motion' in packet for packet in self.output))
        self.assertEqual([command['id'] for command in self.animations()], ['quiet-pose'])
        self.assertFalse(behavior.gesture('jumping', self.sessions))

    def test_work_burst_returns_to_rest_while_tools_continue(self):
        behavior = self.make_behavior({'busy-pose': {'role': 'work', 'frames': 2, 'fps': 2}})
        self.start()
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'busy-pose')
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(len(self.animations()), 1)
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
        self.assertTrue(self.sessions.working)

    def test_idle_debounce_and_minimum_dwell_preserve_a_brief_work_pose(self):
        behavior = self.make_behavior({'busy-pose': {'role': 'work'}})
        self.start()
        behavior.tick('working', self.sessions)
        self.clock.advance(.2)
        self.sessions.handle({'harness': 'cli', 'event': 'tool_done', 'session': 'one', 'call': 'build'})
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.output[-1], {'state': 'idle'})
        self.assertEqual(self.animations()[-1]['id'], 'busy-pose')
        self.clock.advance(1.1)
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'busy-pose')
        self.clock.advance(.7)
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_error_preempts_gesture_cooldown_and_recovers_to_current_activity(self):
        behavior = self.make_behavior({'hello': {'role': 'greeting'}, 'uh-oh': {'role': 'error'}, 'busy': {'role': 'work'}})
        behavior.tick('idle', self.sessions)
        self.assertGreater(behavior.gesture_after, self.clock())
        self.start()
        behavior.tick('error', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'uh-oh')
        self.assertEqual(self.animations()[-1]['repeats'], 1)
        self.assertEqual(behavior.pending['kind'], 'error')
        self.clock.advance(2)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'busy')

    def test_walking_clamps_targets_and_uses_declared_directions(self):
        behavior = self.make_behavior({'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}}, walk_speed=1000)
        self.rng.walk = True
        behavior.tick('idle', self.sessions)
        self.clock.advance(101)
        behavior.tick('idle', self.sessions)
        motion = self.output[-1]['motion']
        self.assertEqual(motion['x'], 400)
        self.assertEqual(motion['y'], 100)
        self.assertEqual(self.output[-1]['animation']['id'], 'east')
        self.assertEqual(motion['command_id'], self.output[-1]['animation']['command_id'])

    def test_speech_drag_menu_and_pointer_pause_walking_until_resume_delay(self):
        for blocked in ({'speech': True}, {'dragging': True}, {'menu': True}, {'pointer': {'x': 200, 'y': 152}}):
            with self.subTest(blocked=blocked):
                self.output.clear()
                self.clock = Clock()
                self.sessions = Sessions(self.clock)
                behavior = self.make_behavior({'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}}, idle_pause=(.1, .1))
                self.rng.walk = True
                behavior.tick('idle', self.sessions)
                self.clock.advance(3)
                behavior.tick('idle', self.sessions)
                self.assertEqual(behavior.pending['kind'], 'walk')
                behavior.renderer_status(self.status(**blocked))
                behavior.tick('idle', self.sessions)
                self.assertIsNone(behavior.pending)
                self.assertIn({'stop_motion': True}, self.output)
                behavior.renderer_status(self.status())
                behavior.tick('idle', self.sessions)
                self.clock.advance(2.9)
                behavior.tick('idle', self.sessions)
                self.assertIsNone(behavior.pending)
                self.clock.advance(.2)
                behavior.tick('idle', self.sessions)
                self.assertEqual(behavior.pending['kind'], 'walk')

    def test_completion_requires_matching_command_and_accepts_interruption(self):
        behavior = self.make_behavior({'hop': {'role': 'celebrate'}})
        behavior.tick('idle', self.sessions)
        self.clean_completion()
        behavior.gesture('hop', self.sessions)
        command = self.animations()[-1]['command_id']
        for packet in ({'event': 'animation', 'command_id': command + 1, 'result': 'finished'},
                       {'event': 'motion', 'command_id': command, 'result': 'finished'},
                       {'event': 'animation', 'command_id': command, 'result': 'unknown'}):
            behavior.completion(packet)
            self.assertEqual(behavior.pending['id'], command)
        behavior.completion({'event': 'animation', 'command_id': command, 'result': 'interrupted'})
        self.assertIsNone(behavior.pending)
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_motion_completion_ignores_animation_finish_and_stale_command(self):
        behavior = self.make_behavior({'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}})
        self.rng.walk = True
        self.rng.angle = math.pi
        behavior.tick('idle', self.sessions)
        self.clock.advance(101)
        behavior.tick('idle', self.sessions)
        command = behavior.pending['id']
        self.assertEqual(self.animations()[-1]['id'], 'west')
        for packet in ({'event': 'animation', 'command_id': command, 'result': 'finished'},
                       {'event': 'motion', 'command_id': command - 1, 'result': 'interrupted'}):
            behavior.completion(packet)
            self.assertEqual(behavior.pending['id'], command)
        behavior.completion({'event': 'motion', 'command_id': command, 'result': 'finished'})
        self.assertIsNone(behavior.pending)
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_celebration_requires_clean_convergence_and_stops_when_work_resumes(self):
        behavior = self.make_behavior({'hooray': {'role': 'celebrate'}})
        behavior.tick('idle', self.sessions)
        self.start()
        self.assertFalse(behavior.gesture('hooray', self.sessions))
        self.finish()
        self.assertTrue(behavior.gesture('hooray', self.sessions))
        self.start()
        behavior.tick('working', self.sessions)
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_work_gesture_requires_pending_tools_but_attention_can_follow_reasoning(self):
        behavior = self.make_behavior({'busy-pose': {'role': 'work'}, 'ponder': {'role': 'attention'}})
        self.sessions.handle({'harness': 'cli', 'event': 'intent', 'session': 'one', 'goal': 'Plan changes'})
        self.assertTrue(self.sessions.active)
        self.assertFalse(self.sessions.working)
        behavior.tick('idle', self.sessions)
        self.assertFalse(behavior.gesture('busy-pose', self.sessions))
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
        self.assertTrue(behavior.gesture('ponder', self.sessions))
        self.assertEqual(self.animations()[-1]['id'], 'ponder')

    def test_error_gesture_returns_to_rest_after_same_tool_recovery(self):
        behavior = self.make_behavior({'concerned': {'role': 'error', 'frames': 4, 'fps': 2}})
        self.start()
        self.sessions.handle({'harness': 'cli', 'event': 'tool_error', 'session': 'one', 'call': 'build', 'detail': 'compiler failure'})
        self.assertTrue(self.sessions.failed)
        behavior.tick('idle', self.sessions)
        self.assertTrue(behavior.gesture('concerned', self.sessions))
        self.assertEqual(behavior.pending['kind'], 'gesture')
        self.sessions.handle({'harness': 'cli', 'event': 'tool_start', 'session': 'one', 'call': 'retry', 'tool': 'build'})
        self.sessions.handle({'harness': 'cli', 'event': 'tool_done', 'session': 'one', 'call': 'retry'})
        self.assertFalse(self.sessions.failed)
        behavior.tick('idle', self.sessions)
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_gaze_tracks_available_pose_and_returns_to_rest_after_quiet(self):
        behavior = self.make_behavior(gaze=True)
        behavior.renderer_status(self.status(pointer={'x': 200, 'y': 1000}))
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'look-0')
        behavior.renderer_status(self.status(pointer={'x': 1000, 'y': 152}))
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'look-4')
        self.clock.advance(1.3)
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')


if __name__ == '__main__':
    unittest.main()

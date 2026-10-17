import base64
import json
import math
import random
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
        self.repeat_minimum = False
        self.repeat_calls = []

    def random(self):
        return 0 if self.walk else 1

    def uniform(self, low, high):
        return self.angle if (low, high) == (0, 2 * math.pi) else low

    def randint(self, low, high):
        self.repeat_calls.append((low, high))
        return low if self.repeat_minimum else high

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
        defaults = {'idle_pause': (1, 1), 'working_pause': (1, 1), 'naps': False, 'work_burst_seconds': (2, 2), 'walk_seconds': (2, 2)}
        behavior = Behavior(Pack(directory), BehaviorConfig(**(defaults | config)), self.clock, self.rng, self.output.append)
        self.assertTrue(behavior.renderer_status(self.status()))
        return behavior

    def status(self, **values):
        return {'kind': 'renderer', 'event': 'status', 'x': 100, 'y': 100, 'width': 200, 'height': 160,
                'screen': {'x': 0, 'y': 0, 'width': 600, 'height': 400},
                'pointer': {'x': 2000, 'y': 2000},
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

    def advance(self, behavior, seconds, state='idle'):
        for _ in range(seconds):
            self.clock.advance(1)
            behavior.tick(state, self.sessions)

    def wait_for_walk(self, behavior):
        for _ in range(600):
            behavior.tick('idle', self.sessions)
            if behavior.pending and behavior.pending['kind'] == 'walk':
                return next(packet['motion'] for packet in reversed(self.output) if 'motion' in packet)
            self.clock.advance(1)
        self.fail('The inactive pet did not wander')

    def test_gesture_repeats_whole_cycles_with_authored_limit(self):
        behavior = self.make_behavior({'hop': {'role': 'celebrate', 'frames': 5, 'fps': 5, 'max_repeats': 2}})
        self.clean_completion()
        self.assertTrue(behavior.gesture('hop', self.sessions))
        self.assertEqual(self.animations()[-1]['repeats'], 2)
        self.assertEqual(self.rng.repeat_calls, [(1, 2)])
        self.assertEqual(behavior.pending['until'], self.clock() + 3)
        self.assertFalse(behavior.gesture('hop', self.sessions))

    def test_quiet_care_reuses_clips_and_click_satisfies_speech_need(self):
        behavior = self.make_behavior({'jumping': {'role': 'celebrate'}}, roaming=False,
                                      care_interval=(10, 10), care_speech_interval=(20, 20))
        self.advance(behavior, 25)
        jumps = [animation for animation in self.animations() if animation['id'] == 'jumping']
        self.assertEqual(len(jumps), 2)
        self.assertTrue(all(animation['repeats'] == 3 for animation in jumps))
        self.assertEqual(behavior.care_remark, 'exercise')
        behavior.satisfy()
        behavior.react('click', self.sessions)
        self.assertIsNone(behavior.care_remark)
        self.advance(behavior, 9)
        self.assertEqual(len([a for a in self.animations() if a['id'] == 'jumping']), 2)

    def test_work_suppresses_care_and_waiting_only_shows_consternation(self):
        behavior = self.make_behavior({'jumping': {'role': 'celebrate'}, 'waving': {'role': 'greeting'},
                                      'failed': {'role': 'error'}}, roaming=False, care_interval=(10, 10))
        self.start()
        self.advance(behavior, 30, 'working')
        self.assertFalse(any(a['id'] in ('jumping', 'waving', 'failed') for a in self.animations()))
        self.sessions.handle({'harness': 'cli', 'event': 'tool_done', 'session': 'one', 'call': 'build'})
        self.sessions.handle({'harness': 'cli', 'event': 'waiting', 'session': 'one', 'call': 'question'})
        self.advance(behavior, 15)
        self.assertTrue(any(a['id'] == 'failed' for a in self.animations()))
        self.assertIsNone(behavior.care_remark)

    def test_long_idle_stretch_favors_naps_then_jumping_and_waving(self):
        behavior = self.make_behavior({'jumping': {'role': 'celebrate'}, 'waving': {'role': 'greeting'},
                                      'failed': {'role': 'error'}}, roaming=False, naps=True)
        behavior.rng = random.Random(29)
        naps = 0
        for _ in range(24 * 3600):
            sleeping = behavior.sleeping
            self.advance(behavior, 1)
            naps += behavior.sleeping and not sleeping
        counts = {clip: sum(a['id'] == clip for a in self.animations()) for clip in ('jumping', 'waving', 'failed')}
        self.assertGreater(naps, sum(counts.values()))
        self.assertGreater(counts['jumping'], counts['failed'])
        self.assertGreater(counts['waving'], counts['failed'])
        self.assertGreater(counts['failed'], 0)

    def test_ninety_idle_minutes_include_each_care_activity(self):
        behavior = self.make_behavior({'jumping': {'role': 'celebrate'}, 'waving': {'role': 'greeting'},
                                      'failed': {'role': 'error'}}, roaming=False, naps=True)
        self.advance(behavior, 90 * 60)
        clips = [a['id'] for a in self.animations() if a['id'] in ('jumping', 'waving', 'failed')]
        self.assertGreaterEqual(clips.count('jumping'), 3)
        self.assertGreaterEqual(clips.count('waving'), 3)
        self.assertGreaterEqual(clips.count('failed'), 1)
        self.assertEqual(clips[:5].count('jumping'), 2)
        self.assertEqual(clips[:5].count('waving'), 2)
        self.assertEqual(clips[:5].count('failed'), 1)

    def test_computer_sleep_and_interactions_defer_care(self):
        behavior = self.make_behavior({'jumping': {'role': 'celebrate'}}, roaming=False, care_interval=(10, 10))
        self.advance(behavior, 5)
        behavior.suspend('system_sleep')
        self.clock.advance(3600)
        behavior.resume()
        self.advance(behavior, 9)
        self.assertFalse(any(a['id'] == 'jumping' for a in self.animations()))
        behavior.renderer_status(self.status(speech=True))
        self.advance(behavior, 20)
        self.assertFalse(any(a['id'] == 'jumping' for a in self.animations()))
        behavior.renderer_status(self.status())
        self.advance(behavior, 5)
        self.assertTrue(any(a['id'] == 'jumping' for a in self.animations()))

    def test_repeat_budget_keeps_long_clips_complete(self):
        behavior = self.make_behavior({'hop': {'role': 'celebrate', 'frames': 10, 'fps': 2}},
                                      repeat_range=(2, 9), repeat_budget=4)
        self.clean_completion()
        self.assertTrue(behavior.gesture('hop', self.sessions))
        self.assertEqual(self.animations()[-1]['repeats'], 1)
        self.clock.advance(4)
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'hop')

    def test_rest_only_pack_stays_still_and_rejects_unavailable_gestures(self):
        behavior = self.make_behavior()
        behavior.tick('idle', self.sessions)
        self.advance(behavior, 600)
        self.assertFalse(any('motion' in packet for packet in self.output))
        self.assertEqual({command['id'] for command in self.animations()}, {'quiet-pose'})
        self.assertFalse(behavior.gesture('jumping', self.sessions))

    def test_work_bursts_commit_then_rest_and_resume_during_an_active_turn(self):
        behavior = self.make_behavior({'busy': {'role': 'work', 'frames': 2, 'fps': 2}})
        self.start()
        behavior.tick('working', self.sessions)
        command = self.animations()[-1]
        self.assertEqual(command['id'], 'busy')
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations(), [command])
        behavior.completion({'event': 'animation', 'command_id': command['command_id'], 'result': 'finished'})
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'busy')
        self.assertTrue(self.sessions.active)

    def test_missing_animation_feedback_expires_after_whole_cycles(self):
        behavior = self.make_behavior({'busy': {'role': 'work', 'frames': 2, 'fps': 2}})
        self.start()
        behavior.tick('working', self.sessions)
        start = self.animations()[-1]
        behavior.completion({'event': 'animation', 'command_id': start['command_id'], 'result': 'finished'})
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.clock.advance(2.9)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'busy')
        self.clock.advance(.1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_work_burst_lasts_for_the_sampled_duration_in_whole_cycles(self):
        behavior = self.make_behavior({'busy': {'role': 'work', 'frames': 41, 'fps': 50}},
                                      work_burst_seconds=(12, 25))
        self.rng.repeat_minimum = True
        self.start()
        behavior.tick('working', self.sessions)
        start = self.animations()[-1]
        behavior.completion({'event': 'animation', 'command_id': start['command_id'], 'result': 'finished'})
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        burst = self.animations()[-1]
        self.assertEqual((burst['id'], burst['repeats']), ('busy', 15))
        self.advance(behavior, 11, 'working')
        self.assertEqual(self.animations()[-1], burst)

    def test_work_burst_honors_the_pack_repeat_limit(self):
        behavior = self.make_behavior({'busy': {'role': 'work', 'frames': 8, 'fps': 10, 'max_repeats': 3}},
                                      work_burst_seconds=(12, 25))
        self.start()
        behavior.tick('working', self.sessions)
        start = self.animations()[-1]
        behavior.completion({'event': 'animation', 'command_id': start['command_id'], 'result': 'finished'})
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['repeats'], 3)

    def test_long_work_walks_between_bursts_and_resumes_focus(self):
        behavior = self.make_behavior({'busy': {'role': 'work'},
                                       'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}})
        self.start()
        for _ in range(600):
            behavior.tick('working', self.sessions)
            if behavior.pending and behavior.pending['kind'] == 'walk':
                break
            self.clock.advance(1)
        else:
            self.fail('The working pet did not walk')
        self.assertTrue(self.sessions.working)
        self.assertGreaterEqual(self.clock() - 100, 225)
        command = behavior.pending['id']
        behavior.completion({'event': 'motion', 'command_id': command, 'result': 'finished'})
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'busy')

    def test_reactions_preempt_and_remain_visible_during_other_work(self):
        behavior = self.make_behavior({'hello': {'role': 'greeting'}, 'uh-oh': {'role': 'error'},
                                       'hop': {'role': 'celebrate'}, 'busy': {'role': 'work'}})
        behavior.gesture('hello', self.sessions)
        self.start()
        behavior.react('trouble')
        self.assertEqual(self.animations()[-1]['id'], 'uh-oh')
        self.assertEqual(self.animations()[-1]['repeats'], 3)
        behavior.react('recovery')
        self.assertEqual(self.animations()[-1]['id'], 'hop')
        self.assertEqual(self.animations()[-1]['repeats'], 3)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'hop')
        self.assertTrue(self.sessions.active)

    def test_short_reaction_plays_complete_cycles_for_its_visible_duration(self):
        behavior = self.make_behavior({'hop': {'role': 'celebrate', 'frames': 8, 'fps': 10}})
        self.assertTrue(behavior.react('success'))
        animation = self.animations()[-1]
        self.assertEqual(animation['repeats'], 4)
        self.assertAlmostEqual(animation['repeats'] * .8, 3.2)
        self.clock.advance(3.2)
        behavior.tick('idle', self.sessions)
        self.assertEqual(behavior.pending['id'], animation['command_id'])
        self.assertEqual(self.animations()[-1]['id'], 'hop')
        behavior.completion({'event': 'animation', 'command_id': animation['command_id'], 'result': 'finished'})
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_each_named_reaction_remains_visible_for_short_animation_cycles(self):
        for kind, clip, role in (
            ('start', 'review', 'attention'), ('investigation', 'review', 'attention'),
            ('waiting', 'waiting', 'attention'), ('trouble', 'failed', 'error'),
            ('failure', 'failed', 'error'), ('recovery', 'jumping', 'celebrate'),
            ('success', 'jumping', 'celebrate'), ('cancelled', 'review', 'attention'),
            ('click', 'waving', 'greeting'),
        ):
            with self.subTest(reaction=kind):
                behavior = self.make_behavior({clip: {'role': role, 'fps': 5}})
                self.assertTrue(behavior.react(kind))
                animation = self.animations()[-1]
                self.assertEqual(animation['id'], clip)
                self.assertEqual(animation['repeats'], 13)
                self.assertAlmostEqual(animation['repeats'] * .2, 2.6)

    def test_reaction_duration_respects_the_authored_repeat_limit(self):
        behavior = self.make_behavior({'hop': {'role': 'celebrate', 'frames': 8, 'fps': 10, 'max_repeats': 2}})
        self.assertTrue(behavior.react('success'))
        self.assertEqual(self.animations()[-1]['repeats'], 2)

    def test_long_reaction_plays_one_complete_cycle(self):
        behavior = self.make_behavior({'hop': {'role': 'celebrate', 'frames': 10, 'fps': 2}})
        self.assertTrue(behavior.react('success'))
        animation = self.animations()[-1]
        self.assertEqual(animation['repeats'], 1)
        self.clock.advance(2.5)
        behavior.tick('idle', self.sessions)
        self.assertEqual(behavior.pending['id'], animation['command_id'])
        self.assertEqual(self.animations()[-1]['id'], 'hop')
        self.clock.advance(2.5)
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'hop')
        behavior.completion({'event': 'animation', 'command_id': animation['command_id'], 'result': 'finished'})
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_investigation_and_input_wait_use_attention_poses(self):
        behavior = self.make_behavior({'waiting': {'role': 'attention'}, 'review': {'role': 'attention'},
                                       'busy': {'role': 'work'}})
        self.sessions.handle({'harness': 'cli', 'event': 'tool_start', 'session': 'one',
                              'call': 'inspect', 'tool': 'Read file'})
        self.assertTrue(behavior.attention(self.sessions, 'cli:one'))
        self.assertEqual(self.animations()[-1]['id'], 'review')
        self.sessions.handle({'harness': 'cli', 'event': 'waiting', 'session': 'one', 'call': 'question'})
        self.assertTrue(behavior.attention(self.sessions, 'cli:one'))
        self.assertEqual(self.animations()[-1]['id'], 'waiting')
        self.assertFalse(behavior.gesture('busy', self.sessions))

    def test_walk_waits_for_matching_motion_feedback(self):
        behavior = self.make_behavior({'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}})
        self.rng.angle = math.pi
        motion = self.wait_for_walk(behavior)
        self.assertEqual(self.animations()[-1]['id'], 'west')
        command = motion['command_id']
        for packet in ({'event': 'animation', 'command_id': command, 'result': 'finished'},
                       {'event': 'animation', 'command_id': command, 'result': 'interrupted'},
                       {'event': 'motion', 'command_id': command - 1, 'result': 'interrupted'},
                       {'event': 'motion', 'command_id': command, 'result': 'unknown'}):
            behavior.completion(packet)
            self.assertEqual(behavior.pending['id'], command)
        behavior.completion({'event': 'motion', 'command_id': command, 'result': 'finished'})
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_missing_motion_feedback_returns_to_rest_at_deadline(self):
        behavior = self.make_behavior({'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}})
        motion = self.wait_for_walk(behavior)
        self.clock.advance(motion['seconds'] + .9)
        behavior.tick('idle', self.sessions)
        self.assertIsNotNone(behavior.pending)
        self.clock.advance(.1)
        behavior.tick('idle', self.sessions)
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_stale_feedback_cannot_finish_a_new_reaction(self):
        behavior = self.make_behavior({'hop': {'role': 'celebrate'}, 'hello': {'role': 'greeting'}})
        behavior.react('success')
        command = self.animations()[-1]['command_id']
        behavior.react('click')
        current = self.animations()[-1]
        behavior.completion({'event': 'animation', 'command_id': command, 'result': 'finished'})
        self.assertEqual(behavior.pending['id'], current['command_id'])
        behavior.completion({'event': 'motion', 'command_id': current['command_id'], 'result': 'interrupted'})
        self.assertEqual(behavior.pending['id'], current['command_id'])
        behavior.completion({'event': 'animation', 'command_id': current['command_id'], 'result': 'interrupted'})
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_walk_targets_stay_near_home_and_inside_screen(self):
        behavior = self.make_behavior({'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}}, walk_speed=1000)
        motion = self.wait_for_walk(behavior)
        self.assertLessEqual(math.hypot(motion['x'] - 100, motion['y'] - 100), 160)
        self.assertEqual(motion['x'], 260)
        self.assertEqual(self.animations()[-1]['id'], 'east')
        behavior.renderer_status(self.status(dragging=True, x=390, y=235))
        behavior.tick('idle', self.sessions)
        behavior.renderer_status(self.status(x=390, y=235))
        motion = self.wait_for_walk(behavior)
        self.assertEqual(motion['x'], 400)
        self.assertLessEqual(math.hypot(motion['x'] - 390, motion['y'] - 235), 160)
        self.assertLessEqual(motion['y'], 240)

    def test_walk_duration_honors_authored_repeat_limit(self):
        behavior = self.make_behavior({'west': {'role': 'walk-left', 'max_repeats': 1},
                                       'east': {'role': 'walk-right', 'max_repeats': 1}},
                                      walk_seconds=(6, 6))
        motion = self.wait_for_walk(behavior)
        self.assertEqual(motion['seconds'], 1)
        self.assertEqual(self.animations()[-1]['repeats'], 1)
        self.assertLessEqual(math.hypot(motion['x'] - 100, motion['y'] - 100), 40)

    def test_investigation_focus_prefers_review_over_waiting(self):
        behavior = self.make_behavior({'waiting': {'role': 'attention'}, 'review': {'role': 'attention'}})
        self.sessions.handle({'harness': 'cli', 'event': 'tool_start', 'session': 'one',
                              'call': 'inspect', 'tool': 'Read file'})
        behavior.tick('working', self.sessions)
        start = self.animations()[-1]
        behavior.completion({'event': 'animation', 'command_id': start['command_id'], 'result': 'finished'})
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'review')
        self.assertEqual(behavior.pending['kind'], 'focus')

    def test_interaction_stops_walk_and_delays_resume(self):
        for blocked in ({'speech': True}, {'dragging': True}, {'menu': True}, {'pointer': {'x': 200, 'y': 152}}):
            with self.subTest(blocked=blocked):
                behavior = self.make_behavior({'west': {'role': 'walk-left'}, 'east': {'role': 'walk-right'}})
                motion = self.wait_for_walk(behavior)
                behavior.renderer_status(self.status(**blocked))
                behavior.tick('idle', self.sessions)
                self.assertIsNone(behavior.pending)
                self.assertIn({'stop_motion': True}, self.output)
                behavior.completion({'event': 'motion', 'command_id': motion['command_id'], 'result': 'finished'})
                behavior.renderer_status(self.status())
                behavior.tick('idle', self.sessions)
                self.advance(behavior, 2)
                self.assertFalse(behavior.pending and behavior.pending['kind'] == 'walk')

    def test_computer_sleep_suspends_actions_until_resume(self):
        behavior = self.make_behavior({'busy': {'role': 'work'}})
        self.start()
        behavior.tick('working', self.sessions)
        behavior.suspend('computer_sleep')
        self.advance(behavior, 600, 'working')
        self.assertIsNone(behavior.pending)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
        behavior.resume()
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'busy')

    def test_idle_pet_stays_quiet_when_care_is_disabled(self):
        behavior = self.make_behavior({'hello': {'role': 'greeting'}}, care=False)
        behavior.tick('idle', self.sessions)
        self.advance(behavior, 600)
        self.assertEqual({command['id'] for command in self.animations()}, {'quiet-pose'})
        behavior.react('click')
        self.assertEqual(self.animations()[-1]['id'], 'hello')

    def test_naps_require_two_quiet_minutes_and_use_rest_animation(self):
        behavior = self.make_behavior({'snooze': {'role': 'sleep'}}, naps=True, nap_interval=(1, 1), nap_seconds=(20, 20))
        behavior.tick('idle', self.sessions)
        self.advance(behavior, 119)
        self.assertFalse(behavior.sleeping)
        self.advance(behavior, 121)
        self.assertTrue(any(packet.get('sleeping') is True for packet in self.output))
        self.assertNotIn('snooze', [command['id'] for command in self.animations()])
        self.assertNotIn('snooze', behavior.pack.gesture_ids)
        self.assertFalse(behavior.gesture('snooze', self.sessions))

    def wait_for_nap(self, behavior):
        for _ in range(300):
            behavior.tick('idle', self.sessions)
            if behavior.sleeping:
                return
            self.clock.advance(1)
        self.fail('The inactive pet did not nap')

    def test_nap_lasts_whole_interval_and_requires_awake_time_before_next_nap(self):
        behavior = self.make_behavior(naps=True, nap_interval=(120, 120), nap_seconds=(20, 20))
        self.wait_for_nap(behavior)
        self.clock.advance(19)
        behavior.tick('idle', self.sessions)
        self.assertTrue(behavior.sleeping)
        self.clock.advance(1)
        behavior.tick('idle', self.sessions)
        self.assertFalse(behavior.sleeping)
        self.advance(behavior, 119)
        self.assertFalse(behavior.sleeping)

    def test_outstanding_tools_and_waiting_turns_keep_pet_awake(self):
        for event in ('tool_start', 'waiting'):
            with self.subTest(event=event):
                self.sessions = Sessions(self.clock)
                self.sessions.handle({'harness': 'cli', 'session': 'one', 'event': event, 'goal': 'Plan changes', 'call': 'long'})
                behavior = self.make_behavior(naps=True, nap_interval=(1, 1))
                self.advance(behavior, 600)
                self.assertFalse(behavior.sleeping)

    def test_quiet_live_turn_settles_and_naps_after_activity_becomes_uncertain(self):
        self.sessions.handle({'harness': 'cli', 'session': 'one', 'event': 'intent'})
        behavior = self.make_behavior(naps=True, nap_interval=(1, 1), nap_seconds=(60, 60))
        self.advance(behavior, 299)
        self.assertTrue(self.sessions.working)
        self.assertFalse(behavior.sleeping)
        self.wait_for_nap(behavior)
        self.assertTrue(self.sessions.active)
        self.assertFalse(self.sessions.working)
        self.assertTrue(behavior.sleeping)
        self.sessions.handle({'harness': 'cli', 'session': 'one', 'event': 'progress'})
        behavior.tick('working', self.sessions)
        self.assertFalse(behavior.sleeping)

    def test_finished_failed_turn_can_settle_and_nap(self):
        self.start()
        self.sessions.handle({'harness': 'cli', 'session': 'one', 'event': 'turn_error'})
        self.assertFalse(self.sessions.active)
        behavior = self.make_behavior(naps=True, nap_interval=(1, 1))
        self.wait_for_nap(behavior)
        self.assertTrue(behavior.sleeping)

    def test_new_work_and_interaction_wake_naps_immediately(self):
        for interaction in (None, {'speech': True}, {'dragging': True}, {'menu': True}, {'pointer': {'x': 200, 'y': 152}}):
            with self.subTest(interaction=interaction):
                self.sessions = Sessions(self.clock)
                behavior = self.make_behavior(naps=True, nap_interval=(1, 1))
                self.wait_for_nap(behavior)
                if interaction is None:
                    self.start()
                else:
                    behavior.renderer_status(self.status(**interaction))
                behavior.tick('working' if interaction is None else 'idle', self.sessions)
                self.assertFalse(behavior.sleeping)
                self.assertIn({'sleeping': False}, self.output)

    def test_pointer_gaze_updates_at_rest_and_preserves_a_work_cycle(self):
        behavior = self.make_behavior({'busy': {'role': 'work'}}, gaze=True)
        behavior.renderer_status(self.status(pointer={'x': 200, 'y': 1000}))
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'look-0')
        behavior.renderer_status(self.status(pointer={'x': 1000, 'y': 152}))
        behavior.tick('idle', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'look-4')
        self.advance(behavior, 4)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
        self.start()
        behavior.tick('working', self.sessions)
        self.advance(behavior, 3, 'working')
        command = self.animations()[-1]
        self.assertEqual(command['id'], 'busy')
        behavior.renderer_status(self.status(pointer={'x': 200, 'y': 1000}))
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1], command)

    def test_interaction_restarts_the_two_minute_quiet_period(self):
        behavior = self.make_behavior(naps=True, nap_interval=(1, 1))
        behavior.tick('idle', self.sessions)
        self.advance(behavior, 119)
        behavior.renderer_status(self.status(speech=True))
        behavior.tick('idle', self.sessions)
        behavior.renderer_status(self.status())
        behavior.tick('idle', self.sessions)
        self.advance(behavior, 119)
        self.assertFalse(behavior.sleeping)

    def test_long_menu_and_drag_restart_quiet_time_when_released(self):
        for interaction in ('menu', 'dragging'):
            with self.subTest(interaction=interaction):
                behavior = self.make_behavior(naps=True, roaming=False, nap_interval=(1, 1))
                behavior.tick('idle', self.sessions)
                self.clock.advance(10)
                behavior.renderer_status(self.status(**{interaction: True}))
                behavior.tick('idle', self.sessions)
                self.clock.advance(250)
                behavior.tick('idle', self.sessions)
                behavior.renderer_status(self.status())
                behavior.tick('idle', self.sessions)
                self.assertFalse(behavior.sleeping)
                self.advance(behavior, 119)
                self.assertFalse(behavior.sleeping)
                self.wait_for_nap(behavior)
                self.assertTrue(behavior.sleeping)

    def test_click_restarts_quiet_time_without_an_available_wave(self):
        behavior = self.make_behavior(naps=True, roaming=False, nap_interval=(1, 1))
        behavior.tick('idle', self.sessions)
        self.advance(behavior, 119)
        behavior.react('click', self.sessions)
        self.advance(behavior, 119)
        self.assertFalse(behavior.sleeping)
        self.wait_for_nap(behavior)
        self.assertTrue(behavior.sleeping)

    def test_missing_review_starts_work_immediately_using_running(self):
        behavior = self.make_behavior({'running': {'role': 'work'}, 'waiting': {'role': 'attention'}},
                                      idle_pause=(60, 60))
        behavior.tick('idle', self.sessions)
        self.start()
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'running')
        self.assertTrue(self.sessions.working)

    def test_missing_running_uses_review_for_work_focus(self):
        behavior = self.make_behavior({'review': {'role': 'attention'}, 'waiting': {'role': 'attention'}})
        self.start()
        behavior.tick('working', self.sessions)
        start = self.animations()[-1]
        behavior.completion({'event': 'animation', 'command_id': start['command_id'], 'result': 'finished'})
        self.clock.advance(1)
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'review')
        self.assertEqual(behavior.pending['kind'], 'focus')

    def test_missing_reaction_uses_one_alternative_then_rest(self):
        cases = (
            ('start', 'running', 'work'), ('investigation', 'running', 'work'),
            ('waiting', 'review', 'attention'), ('trouble', 'review', 'attention'),
            ('failure', 'review', 'attention'), ('recovery', 'waving', 'greeting'),
            ('success', 'waving', 'greeting'), ('click', 'review', 'attention'),
        )
        for kind, alternative, role in cases:
            with self.subTest(kind=kind, alternative=alternative):
                behavior = self.make_behavior({alternative: {'role': role}})
                self.assertTrue(behavior.react(kind))
                self.assertEqual(self.animations()[-1]['id'], alternative)
            with self.subTest(kind=kind, alternative='rest'):
                behavior = self.make_behavior()
                self.assertTrue(behavior.react(kind))
                self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
                self.assertIsNone(self.animations()[-1]['repeats'])

    def test_missing_success_stops_at_rest_when_its_wave_alternative_is_missing(self):
        behavior = self.make_behavior({'review': {'role': 'attention'}, 'running': {'role': 'work'}})
        behavior.react('success')
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')

    def test_missing_work_and_review_keep_rest_as_the_last_resort(self):
        behavior = self.make_behavior({'waiting': {'role': 'attention'}}, idle_pause=(60, 60))
        behavior.tick('idle', self.sessions)
        self.start()
        behavior.tick('working', self.sessions)
        self.assertEqual(self.animations()[-1]['id'], 'quiet-pose')
        self.advance(behavior, 10, 'working')
        self.assertEqual({clip['id'] for clip in self.animations()}, {'quiet-pose'})

    def test_low_priority_reactions_preserve_a_prominent_outcome(self):
        behavior = self.make_behavior({'review': {'role': 'attention'}, 'hop': {'role': 'celebrate'}})
        self.assertTrue(behavior.react('success'))
        command = self.animations()[-1]
        self.assertFalse(behavior.react('investigation'))
        self.assertFalse(behavior.react('success'))
        self.assertEqual(self.animations()[-1], command)

    def test_naps_can_be_disabled(self):
        behavior = self.make_behavior(naps=False, nap_interval=(1, 1))
        self.advance(behavior, 600)
        self.assertFalse(any(packet.get('sleeping') for packet in self.output))


if __name__ == '__main__':
    unittest.main()

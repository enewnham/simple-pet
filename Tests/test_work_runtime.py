import asyncio
import json
from pathlib import Path
import random
import socket
import tempfile
import unittest
import uuid

import httpx
from anthropic import AsyncAnthropic

from pet_daemon.config import Config
from pet_daemon.daemon import Daemon
from pet_daemon.sources.git import command
from Tests.test_runtime import PNG, reply


class WorkRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="pet-work-runtime-")
        root = Path(self.temp.name)
        (root / "pixel.png").write_bytes(PNG)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "pet": "quiet",
                    "rest": "still",
                    "clips": {
                        "still": {"frames": ["pixel.png"], "fps": 1, "role": "rest"},
                    },
                }
            )
        )
        self.config = Config(
            pet_dir=root,
            daemon_socket=str(root / "daemon.sock"),
            overlay_socket=str(root / "overlay.sock"),
            summarize_on=[],
            observation_seconds=(3600, 3600),
            request_timeout=5,
            memory_file=root / "memory.sqlite3",
        )
        self.config.behavior.automatic_speech_cooldown_seconds = 3600
        self.requests = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.slow = False
        self.speech = "The persistence checks need another look."
        self.github_commands = []
        self.related_prs = {}
        self.stack_entries = []
        self.repository_path = root / "repository"
        self.repository_path.mkdir()
        for args in (
            ("init", "-b", "feature"),
            ("config", "user.email", "pet@example.test"),
            ("config", "user.name", "Pet"),
            ("config", "commit.gpgsign", "false"),
        ):
            await command("git", *args, cwd=self.repository_path)
        self.head = await self.commit("initial session state")
        await command(
            "git",
            "remote",
            "add",
            "origin",
            "https://example.test/owner/repository.git",
            cwd=self.repository_path,
        )
        self.pr = {
            "number": 7,
            "url": "https://example.test/owner/repository/pull/7",
            "state": "OPEN",
            "mergedAt": None,
            "title": "Persist coding session context",
            "body": "Retain the requested session context when the daemon restarts.",
            "reviewDecision": "REVIEW_REQUIRED",
            "isDraft": True,
            "headRefOid": self.head,
            "headRefName": "feature",
            "headRepository": {"nameWithOwner": "owner/repository"},
            "headRepositoryOwner": {"login": "owner"},
            "reviewRequests": [],
            "statusCheckRollup": self.checks("FAILURE"),
        }

        async def handler(request):
            self.requests.append(json.loads(request.content))
            speech = self.speech
            self.entered.set()
            if self.slow:
                await self.release.wait()
            return reply(speech)

        self.client = AsyncAnthropic(
            api_key="test-fixture",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        self.overlay = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.overlay.bind(self.config.overlay_socket)
        self.overlay.setblocking(False)
        self.hook = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        await self.start_daemon()

    async def asyncTearDown(self):
        await self.daemon.close()
        await self.client.close()
        self.overlay.close()
        self.hook.close()
        self.temp.cleanup()

    async def commit(self, content):
        (self.repository_path / "session.txt").write_text(content + "\n")
        await command("git", "add", "session.txt", cwd=self.repository_path)
        await command(
            "git", "commit", "-m", "Record session state", cwd=self.repository_path
        )
        return (
            await command("git", "rev-parse", "HEAD", cwd=self.repository_path)
        ).strip()

    def checks(self, result):
        return [
            {
                "name": "Session persistence tests",
                "conclusion": result,
                "headSha": self.head,
                "app": {"slug": "github-actions"},
            }
        ]

    async def start_daemon(self):
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        self.daemon.application.providers[0].run = self.repository_command
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        await self.daemon.start()
        self.send(
            {
                "kind": "renderer",
                "event": "status",
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 160,
                "screen": {"x": 0, "y": 0, "width": 1000, "height": 800},
                "pointer": {"x": 2000, "y": 2000},
                "dragging": False,
                "menu": False,
                "speech": False,
            }
        )
        await self.packet(lambda packet: packet.get("state") == "idle")

    async def repository_command(self, *args, **kwargs):
        if args[0] != "gh":
            return await command(*args, **kwargs)
        self.github_commands.append(args)
        if args[:3] == ("gh", "api", "user"):
            return "elliot"
        if args[:3] == ("gh", "repo", "view"):
            return json.dumps({"isFork": False})
        if args[:3] == ("gh", "pr", "list"):
            return json.dumps([self.pr])
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps(self.related_prs.get(args[3], self.pr))
        if args[:2] == ("gh", "api") and "/stacks?" in args[2]:
            return json.dumps(self.stack_entries)
        self.fail(f"Unexpected GitHub command: {args}")

    def send(self, packet):
        self.hook.sendto(json.dumps(packet).encode(), self.config.daemon_socket)

    async def native_work(self):
        self.send(
            {
                "kind": "event",
                "harness": "codex",
                "session": "persistence",
                "event": "intent",
                "turn": "implementation",
                "repo": str(self.repository_path),
                "goal": "Preserve the coding session context across daemon restarts",
            }
        )
        await self.wait_for(
            lambda: "codex:persistence" in self.daemon.application.sessions.lanes
        )
        self.send(
            {
                "kind": "event",
                "harness": "codex",
                "session": "persistence",
                "event": "turn_done",
                "turn": "implementation",
                "repo": str(self.repository_path),
                "outcome": "success",
                "progress": "Implemented session persistence; validation passed",
            }
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["codex:persistence"].outcome
                == "success"
            )
        )
        self.daemon.application.attention.pending = None
        await self.wait_for(
            lambda: any(
                item.kind == "resolve_checks"
                for item in self.daemon.application.memory.work.opportunities()
            )
        )
        await self.wait_for(lambda: not self.daemon.application.providers[0].tasks)
        self.assertEqual(self.requests, [])
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        await self.drain()
        return next(
            item
            for item in self.daemon.application.memory.work.opportunities()
            if item.kind == "resolve_checks"
        )

    async def native_stack_work(self):
        sibling_url = "https://example.test/owner/repository/pull/8"
        self.related_prs[sibling_url] = {
            **self.pr,
            "number": 8,
            "url": sibling_url,
            "headRefName": "followup",
            "title": "Verify restored sessions",
            "statusCheckRollup": self.checks("SUCCESS"),
        }
        self.stack_entries = [
            {
                "number": 4,
                "pull_requests": [
                    {"number": 7, "state": "open", "draft": True},
                    {"number": 8, "state": "open", "draft": True},
                ],
            }
        ]
        await self.native_work()
        await self.wait_for(
            lambda: any(
                item.pr and item.pr.url == sibling_url
                for item in self.daemon.application.providers[0].snapshot()
            )
        )
        await self.wait_for(lambda: not self.daemon.application.providers[0].tasks)
        opportunity = next(
            item
            for item in self.daemon.application.memory.work.opportunities()
            if item.kind == "resolve_checks"
        )
        return opportunity, sibling_url

    async def merge_sibling(self, sibling_url):
        self.related_prs[sibling_url].update(
            state="MERGED", mergedAt="2026-10-19T12:00:00Z"
        )
        await self.refresh_sources(
            lambda: any(
                item.pr and item.pr.url == sibling_url and item.pr.state == "MERGED"
                for item in self.daemon.application.providers[0].snapshot()
            )
        )

    async def packet(self, predicate):
        async def receive():
            while True:
                data = await asyncio.get_running_loop().sock_recv(self.overlay, 65536)
                packet = json.loads(data)
                if predicate(packet):
                    return packet

        return await asyncio.wait_for(receive(), 5)

    async def wait_for(self, condition):
        async def wait():
            while not condition():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait(), 5)

    async def drain(self):
        packets = []
        while True:
            try:
                packets.append(json.loads(self.overlay.recv(65536)))
            except BlockingIOError:
                return packets

    async def refresh_sources(self, predicate):
        previous = len(self.github_commands)
        self.daemon.application.providers[0].local_interval = 0
        self.daemon.application.providers[0].github_interval = 0
        try:
            await self.wait_for(
                lambda: len(self.github_commands) > previous and predicate()
            )
        finally:
            self.daemon.application.providers[0].local_interval = 3600
            self.daemon.application.providers[0].github_interval = 3600
        await self.wait_for(lambda: not self.daemon.application.providers[0].tasks)

    def schedule_routine(self):
        self.daemon.application.attention.next_observation = self.daemon.clock()
        self.daemon.application.tick()

    def focus(self):
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        return payload["focus_opportunity"]

    async def accept(self, speech):
        self.send(
            {
                "kind": "renderer",
                "event": "speech_accepted",
                "delivery_id": speech["delivery_id"],
            }
        )
        await self.wait_for(
            lambda: (
                (
                    outcome := self.daemon.application.memory.message_outcome(
                        speech["delivery_id"]
                    )
                )
                is not None
                and outcome["outcome"] == "accepted"
            )
        )

    def assert_human_context(self, value):
        if isinstance(value, dict):
            forbidden = {
                "id",
                "revision",
                "head",
                "number",
                "repository",
                "session",
                "work_id",
                "worktree",
                "checkout",
                "staged",
                "unstaged",
                "untracked",
                "ahead",
                "behind",
                "check_revision",
                "count",
                "timestamp",
                "changed_at",
                "version",
                "delivery_id",
            }
            self.assertFalse(forbidden.intersection(value))
            for item in value.values():
                self.assert_human_context(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_human_context(item)

    async def test_native_activity_and_observed_checks_ground_an_acknowledged_remark(
        self,
    ):
        opportunity = await self.native_work()
        self.schedule_routine()
        speech = await self.packet(lambda packet: "delivery_id" in packet)
        focus = self.focus()
        self.assertEqual(
            focus["intent"],
            "Preserve the coding session context across daemon restarts",
        )
        self.assertEqual(focus["kind"], "resolve_checks")
        self.assertEqual(focus["artifact"]["kind"], "pull_request")
        self.assertEqual(focus["artifact"]["title"], "Persist coding session context")
        github = next(
            item
            for item in focus["evidence"]
            if item["source"] == "github" and item["kind"] == "observation"
        )
        self.assertTrue(github["output"]["is_draft"])
        self.assertTrue(github["checks_current"])
        self.assertEqual(github["checks"], [["Session persistence tests", "FAILURE"]])
        progress = next(
            item
            for item in focus["evidence"]
            if item["source"] == "harness" and item["kind"] == "progress"
        )
        self.assertEqual(
            progress["reported_progress"],
            "Implemented session persistence; validation passed",
        )
        self.assertNotIn("execution", progress)
        context = self.daemon.application.memory.work.session_context(
            "codex:persistence"
        )
        execution = next(
            item["data"] for item in context["activity"] if item["kind"] == "execution"
        )
        self.assertEqual(execution["outcome"], "success")
        self.assertTrue(focus["uncertainty"])
        self.assertEqual(
            {item["kind"] for item in context["artifacts"]}, {"branch", "pull_request"}
        )
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertNotIn("activity", payload)
        self.assertIsNone(payload["focus_session"])
        self.assert_human_context(focus)
        encoded = json.dumps(focus)
        for identifier in (self.head, "codex:persistence", str(self.repository_path)):
            self.assertNotIn(identifier, encoded)
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.send(
            {
                "kind": "renderer",
                "event": "speech_accepted",
                "delivery_id": str(uuid.uuid4()),
            }
        )
        await asyncio.sleep(0)
        self.assertEqual(self.daemon.application.memory.speech(), [])
        await self.accept(speech)
        self.assertFalse(
            self.daemon.application.memory.work.is_current(
                opportunity.id, opportunity.version
            )
        )
        self.assertEqual(
            self.daemon.application.memory.speech()[0]["reason"], "routine"
        )
        await self.refresh_sources(
            lambda: any(
                item.pr for item in self.daemon.application.providers[0].snapshot()
            )
        )
        self.assertFalse(
            self.daemon.application.memory.work.is_current(
                opportunity.id, opportunity.version
            )
        )
        self.schedule_routine()
        await asyncio.sleep(0.06)
        self.assertEqual(len(self.requests), 1)
        await self.daemon.close()
        await self.start_daemon()
        self.assertFalse(
            self.daemon.application.memory.work.is_current(
                opportunity.id, opportunity.version
            )
        )
        self.assertEqual(
            self.daemon.application.memory.speech()[0]["text"], speech["text"]
        )

    async def test_new_head_reassesses_checks_and_green_checks_require_an_explicit_review_request(
        self,
    ):
        original = await self.native_work()
        self.schedule_routine()
        await self.accept(await self.packet(lambda packet: "delivery_id" in packet))
        self.head = await self.commit("persist session context during restart")
        self.pr.update(headRefOid=self.head, statusCheckRollup=self.checks("FAILURE"))
        await self.refresh_sources(
            lambda: any(
                item.kind == "resolve_checks"
                and item.version > original.version
                and item.artifact.get("revision") == self.head
                for item in self.daemon.application.memory.work.opportunities()
            )
        )
        current = next(
            item
            for item in self.daemon.application.memory.work.opportunities()
            if item.kind == "resolve_checks"
        )
        self.assertEqual(current.id, original.id)
        self.assertEqual(current.artifact["revision"], self.head)
        self.speech = "The revised session storage still has a failing check."
        self.schedule_routine()
        await self.accept(await self.packet(lambda packet: "delivery_id" in packet))
        self.assertEqual(self.focus()["kind"], "resolve_checks")
        self.pr["statusCheckRollup"] = self.checks("SUCCESS")
        await self.refresh_sources(
            lambda: any(
                item.pr and item.pr.checks and item.pr.checks[0][1] == "SUCCESS"
                for item in self.daemon.application.providers[0].snapshot()
            )
        )
        self.assertFalse(
            any(
                item.kind == "resolve_checks"
                for item in self.daemon.application.memory.work.opportunities()
            )
        )
        self.assertFalse(
            any(
                item.kind == "review_output"
                for item in self.daemon.application.memory.work.opportunities()
            )
        )
        self.pr["reviewRequests"] = [{"login": "elliot"}]
        await self.refresh_sources(
            lambda: any(
                item.kind == "review_output"
                for item in self.daemon.application.memory.work.opportunities()
            )
        )
        self.speech = "There is a requested review on the session persistence change."
        self.schedule_routine()
        await self.accept(await self.packet(lambda packet: "delivery_id" in packet))
        focus = self.focus()
        self.assertEqual(focus["kind"], "review_output")
        github = next(
            item
            for item in focus["evidence"]
            if item["source"] == "github" and item["kind"] == "observation"
        )
        self.assertTrue(github["output"]["review_requested"])

    async def test_merge_rejects_pending_generation_without_concluding_the_intended_work(
        self,
    ):
        await self.native_work()
        self.slow = True
        self.schedule_routine()
        await asyncio.wait_for(self.entered.wait(), 5)
        self.pr.update(state="MERGED", mergedAt="2026-10-18T12:00:00Z")
        await self.refresh_sources(
            lambda: any(
                item.pr and item.pr.state == "MERGED"
                for item in self.daemon.application.providers[0].snapshot()
            )
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.assertFalse(
            any(
                item.kind == "resolve_checks"
                for item in self.daemon.application.memory.work.opportunities()
            )
        )
        work = self.daemon.application.memory.db.execute(
            "SELECT intent,status FROM work_items WHERE pet=?",
            (self.daemon.application.memory.pet,),
        ).fetchall()
        self.assertEqual(
            [(item["intent"], item["status"]) for item in work],
            [("Preserve the coding session context across daemon restarts", "active")],
        )

    async def test_unselected_pull_request_metadata_preserves_supported_generation(
        self,
    ):
        original = await self.native_work()
        self.slow = True
        self.schedule_routine()
        await asyncio.wait_for(self.entered.wait(), 5)
        self.pr["body"] = "Document the manual restart validation procedure."
        await self.refresh_sources(
            lambda: any(
                item.pr and item.pr.body == self.pr["body"]
                for item in self.daemon.application.providers[0].snapshot()
            )
        )
        current = self.daemon.application.memory.work.opportunity(original.id)
        self.assertGreater(current.version, original.version)
        self.assertEqual(current.artifact["revision"], original.artifact["revision"])
        self.release.set()
        speech = await self.packet(lambda packet: "delivery_id" in packet)
        await self.accept(speech)
        outcome = self.daemon.application.memory.message_outcome(speech["delivery_id"])
        self.assertEqual(
            (outcome["opportunity_id"], outcome["version"]),
            (original.id, original.version),
        )
        self.assertNotIn(self.pr["body"], json.dumps(self.focus()))
        self.assertTrue(
            self.daemon.application.memory.work.is_current(current.id, current.version)
        )

    async def test_sibling_merge_preserves_generation_supported_by_primary_checks(self):
        original, sibling_url = await self.native_stack_work()
        self.slow = True
        self.schedule_routine()
        await asyncio.wait_for(self.entered.wait(), 5)
        await self.merge_sibling(sibling_url)
        current = self.daemon.application.memory.work.opportunity(original.id)
        self.assertEqual(current.id, original.id)
        self.assertGreater(current.version, original.version)
        self.assertEqual(current.artifact["revision"], original.artifact["revision"])
        self.release.set()
        speech = await self.packet(lambda packet: "delivery_id" in packet)
        await self.accept(speech)
        outcome = self.daemon.application.memory.message_outcome(speech["delivery_id"])
        self.assertEqual(
            (outcome["opportunity_id"], outcome["version"]),
            (original.id, original.version),
        )
        self.assertTrue(
            self.daemon.application.memory.work.is_current(current.id, current.version)
        )

    async def test_sibling_merge_preserves_delivery_retry_and_captured_acceptance(self):
        original, sibling_url = await self.native_stack_work()
        self.schedule_routine()
        first = await self.packet(lambda packet: "delivery_id" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        await self.merge_sibling(sibling_url)
        current = self.daemon.application.memory.work.opportunity(original.id)
        self.assertGreater(current.version, original.version)
        now = self.daemon.clock() + 5
        self.daemon.application.delivery.clock = lambda: now
        self.daemon.application.tick()
        retry = await self.packet(lambda packet: "delivery_id" in packet)
        self.assertEqual(retry, first)
        await self.accept(retry)
        outcome = self.daemon.application.memory.message_outcome(first["delivery_id"])
        self.assertEqual(
            (outcome["opportunity_id"], outcome["version"]),
            (original.id, original.version),
        )
        self.assertEqual(len(self.daemon.application.memory.speech()), 1)
        self.assertTrue(
            self.daemon.application.memory.work.is_current(current.id, current.version)
        )

    async def test_sleep_cancels_retries_and_late_acceptance_records_the_visible_message(
        self,
    ):
        opportunity = await self.native_work()
        self.schedule_routine()
        first = await self.packet(lambda packet: "delivery_id" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.send({"kind": "renderer", "event": "system_sleep"})
        await self.wait_for(lambda: self.daemon.application.suspended)
        now = self.daemon.clock() + 5
        self.daemon.application.delivery.clock = lambda: now
        self.daemon.application.tick()
        self.assertFalse(any("delivery_id" in packet for packet in await self.drain()))
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.send({"kind": "renderer", "event": "system_wake"})
        await self.wait_for(lambda: not self.daemon.application.suspended)
        await self.wait_for(lambda: not self.daemon.application.providers[0].tasks)
        self.daemon.application.tick()
        self.assertFalse(any("delivery_id" in packet for packet in await self.drain()))
        self.assertEqual(len(self.requests), 1)
        await self.accept(first)
        outcome = self.daemon.application.memory.message_outcome(first["delivery_id"])
        self.assertEqual(
            (outcome["opportunity_id"], outcome["version"]),
            (opportunity.id, opportunity.version),
        )


if __name__ == "__main__":
    unittest.main()

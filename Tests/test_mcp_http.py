import asyncio
import os
import socket
import stat
import sys
import tempfile
import unittest
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2 as httpx
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from pet_daemon.mcp import WorkContext, create_server
from pet_daemon.mcp_http import http_application, load_token
from pet_daemon.storage import Memory


class McpHttpTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.memory = Memory(self.root / "memory.sqlite3", "fixture")
        self.context = WorkContext(self.root / "memory.sqlite3", "fixture")
        self.token_file = self.root / "state/pet-mcp.token"

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    @asynccontextmanager
    async def running_server(
        self,
        token: str,
        write_token: str | None = None,
    ) -> AsyncGenerator[str]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            application = http_application(
                self.context, token, "127.0.0.1", port, write_token
            )
            config = uvicorn.Config(
                application, access_log=False, log_level="warning", proxy_headers=False
            )
            server = uvicorn.Server(config)
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        if task.done():
                            await task
                        await asyncio.sleep(0.01)
                yield f"http://127.0.0.1:{port}/mcp"
            finally:
                server.should_exit = True
                async with asyncio.timeout(10):
                    await task

    async def test_http_requires_bearer_before_processing_any_mcp_method(self) -> None:
        token = load_token(self.token_file)
        async with self.running_server(token) as url, httpx.AsyncClient() as client:
            for method in ("GET", "POST", "DELETE"):
                for headers in ({}, {"Authorization": "Bearer invalid"}):
                    with self.subTest(method=method, headers=bool(headers)):
                        response = await client.request(
                            method, url, headers=headers, content=b"invalid body"
                        )
                        self.assertEqual(response.status_code, 401)
                        self.assertIn("Bearer", response.headers["www-authenticate"])
                        self.assertNotIn(token, response.text)
            response = await client.post(f"{url}?token={token}")
            self.assertEqual(response.status_code, 401)

    async def test_authenticated_http_client_lists_and_calls_read_tools(self) -> None:
        token = load_token(self.token_file)
        async with (
            self.running_server(token) as url,
            httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"}
            ) as http_client,
            streamable_http_client(url, http_client=http_client) as streams,
            ClientSession(streams[0], streams[1]) as client,
        ):
            await client.initialize()
            catalog = await client.list_tools()
            self.assertEqual(
                {tool.name for tool in catalog.tools},
                {
                    "mark_workstream",
                    "unmark_workstream",
                    "workstream_mark_history",
                    "record_presence",
                    "workstreams",
                    "workstream",
                    "since",
                    "sessions",
                    "attention",
                    "decisions",
                    "current_work_context",
                    "context_search",
                    "read_passage",
                    "relevant_changes",
                },
            )
            result = await client.call_tool("current_work_context", {})
            assert isinstance(result, CallToolResult)
            self.assertFalse(result.is_error)
            assert result.structured_content is not None
            self.assertEqual(result.structured_content["pet"], "fixture")
            self.assertEqual(result.structured_content["sources"], [])

    async def test_read_token_can_read_but_cannot_write(self) -> None:
        read_token = load_token(self.token_file)
        write_token = load_token(self.root / "writer.token")
        async with (
            self.running_server(read_token, write_token) as url,
            httpx.AsyncClient(
                headers={"Authorization": f"Bearer {read_token}"}
            ) as http_client,
            streamable_http_client(url, http_client=http_client) as streams,
            ClientSession(streams[0], streams[1]) as client,
        ):
            await client.initialize()
            self.assertFalse((await client.call_tool("sessions", {})).is_error)
            calls = {
                "record_presence": {"status": "available"},
                "mark_workstream": {"id": "unknown", "status": "done"},
                "unmark_workstream": {"id": "unknown"},
            }
            for tool, args in calls.items():
                result = await client.call_tool(
                    tool, {"author": "caller", "reason": "test", **args}
                )
                self.assertTrue(result.is_error)
                self.assertIn("cannot write", str(result.content))
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            http_application(self.context, read_token, "127.0.0.1", 8765, read_token)

    async def test_write_token_marks_a_thread_and_records_its_principal(
        self,
    ) -> None:
        from pet_daemon.domain import SessionSource
        from pet_daemon.identity import SessionIdentity
        from pet_daemon.work.models import IntentEvidence, ReportContext, SessionReport
        from pet_daemon.workstream_marks import MarkStore, marks_path, thread_id

        with self.memory.transaction():
            self.memory._work_repository.append_report(
                SessionReport(
                    payload=IntentEvidence(
                        session=SessionIdentity(
                            source=SessionSource.CODEX, native_id="one"
                        ),
                        text="Configure editor",
                    ),
                    context=ReportContext(checkout="/editor"),
                    observed_at=100,
                )
            )
        token = load_token(self.root / "writer.token")
        id = thread_id("/editor")
        args = {"id": id, "author": "same-label", "reason": "Track verification"}
        local_note = {
            "kind": "handoff",
            "sender": {"name": "Builder"},
            "to": {"name": "Reviewer"},
            "work": "local verification",
        }
        remote_note = {**local_note, "work": "remote verification"}
        async with self.running_server(load_token(self.token_file), token) as url:
            async with httpx.AsyncClient() as anonymous:
                rejected = await anonymous.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "mark_workstream",
                            "arguments": {
                                "id": thread_id("/editor"),
                                "author": "remote-companion",
                                "reason": "Focus",
                                "priority": "focus",
                            },
                        },
                    },
                )
                self.assertEqual(rejected.status_code, 401)
                self.assertFalse(marks_path(self.context.path).exists())
            local = create_server(self.context)
            saved = await local.call_tool(
                "mark_workstream", {**args, "notes": [local_note]}
            )
            assert isinstance(saved, CallToolResult)
            self.assertFalse(saved.is_error, saved.content)
            async with (
                httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {token}"}
                ) as http_client,
                streamable_http_client(url, http_client=http_client) as streams,
                ClientSession(streams[0], streams[1]) as client,
            ):
                await client.initialize()
                result = await client.call_tool(
                    "mark_workstream",
                    {**args, "priority": "focus", "notes": [remote_note]},
                )
                assert isinstance(result, CallToolResult)
                self.assertFalse(result.is_error, result.content)
                result = await client.call_tool(
                    "record_presence",
                    {
                        "status": "available",
                        "author": "remote-companion",
                        "reason": "Saw a click",
                    },
                )
                self.assertFalse(result.is_error, result.content)
                self.assertFalse((await client.call_tool("sessions", {})).is_error)
                replacement = {**remote_note, "work": "updated verification"}
                operations = (
                    ("mark_workstream", {"notes": [replacement]}, replacement),
                    ("mark_workstream", {"notes": []}, None),
                    ("mark_workstream", {"notes": [remote_note]}, remote_note),
                    ("unmark_workstream", {"fields": ["notes"]}, None),
                    ("mark_workstream", {"notes": [remote_note]}, remote_note),
                    ("unmark_workstream", {}, None),
                )
                for tool, fields, remote in operations:
                    with self.subTest(tool=tool, fields=fields):
                        result = await client.call_tool(tool, {**args, **fields})
                        self.assertFalse(result.is_error, result.content)
                        read = await client.call_tool("workstream", {"id": id})
                        assert read.structured_content is not None
                        entries = read.structured_content["mark"]["fields"]["notes"][
                            "entries"
                        ]
                        self.assertEqual(
                            {
                                entry["provenance"]["principal"]: entry["value"]
                                for entry in entries
                            },
                            {"local-stdio": local_note}
                            | ({"bearer:pet-mcp-writer": remote} if remote else {}),
                        )
                        self.assertTrue(
                            all(
                                entry["provenance"]["author"] == args["author"]
                                for entry in entries
                            )
                        )
        events = MarkStore(self.context.path, "fixture").history(id)
        self.assertEqual(events[0]["principal"], "bearer:pet-mcp-writer")
        self.assertEqual(events[0]["author"], args["author"])

    async def test_http_rejects_untrusted_host_and_origin(self) -> None:
        token = load_token(self.token_file)
        async with self.running_server(token) as url, httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json, text/event-stream",
            }
            host_response = await client.get(
                url, headers={**headers, "Host": "external.example"}
            )
            self.assertEqual(host_response.status_code, 421)
            origin_response = await client.get(
                url, headers={**headers, "Origin": "https://external.example"}
            )
            self.assertEqual(origin_response.status_code, 403)
        with self.assertRaisesRegex(ValueError, "loopback"):
            http_application(self.context, token, "0.0.0.0", 8765)

    async def test_token_creation_retains_a_private_stable_secret(self) -> None:
        token = load_token(self.token_file)
        self.assertGreaterEqual(len(token), 32)
        self.assertEqual(load_token(self.token_file), token)
        metadata = self.token_file.stat()
        self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
        self.assertEqual(metadata.st_uid, os.getuid())
        self.assertEqual(list(self.token_file.parent.iterdir()), [self.token_file])

    async def test_token_file_rejects_insecure_permissions_and_symlinks(self) -> None:
        load_token(self.token_file)
        self.token_file.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "0600"):
            load_token(self.token_file)
        self.token_file.chmod(0o600)
        link = self.root / "token-link"
        link.symlink_to(self.token_file)
        with self.assertRaises(OSError):
            load_token(link)
        directory = self.root / "token-directory"
        directory.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError, "regular file"):
            load_token(directory)

    async def test_http_cli_keeps_bearer_tokens_out_of_process_output(self) -> None:
        token = load_token(self.token_file)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pet_daemon.mcp",
            "--http",
            "--port",
            str(port),
            "--memory-file",
            str(self.root / "memory.sqlite3"),
            "--pet",
            "fixture",
            "--token-file",
            str(self.token_file),
            cwd=Path(__file__).resolve().parents[1],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            url = f"http://127.0.0.1:{port}/mcp"
            async with httpx.AsyncClient() as client, asyncio.timeout(10):
                while True:
                    try:
                        response = await client.get(url)
                        break
                    except httpx.ConnectError:
                        if process.returncode is not None:
                            self.fail(
                                "MCP HTTP process exited before accepting requests"
                            )
                        await asyncio.sleep(0.01)
                self.assertEqual(response.status_code, 401)
                response = await client.get(f"{url}?token={token}")
                self.assertEqual(response.status_code, 401)
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    content=b"invalid body",
                )
                self.assertNotEqual(response.status_code, 401)
        finally:
            if process.returncode is None:
                process.terminate()
            async with asyncio.timeout(10):
                stdout, stderr = await process.communicate()
        self.assertNotIn(token.encode(), stdout)
        self.assertNotIn(token.encode(), stderr)

    async def test_token_file_accepts_standard_bearer_characters(self) -> None:
        load_token(self.token_file)
        token = "Abcdefghijklmnopqrstuvwxyz0123456789-._~+/=="
        self.token_file.write_text(token + "\n")
        self.assertEqual(load_token(self.token_file), token)

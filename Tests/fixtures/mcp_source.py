import asyncio
import sys

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("source-fixture", log_level="ERROR")


@server.tool(annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False))
def search(query: str = "", id: str = "fixture-issue") -> dict[str, object]:
    return {
        "issue": {
            "id": id,
            "identifier": "PET-123",
            "title": "Preserve rollback ownership",
            "description": "Decision: confirm the owner before rollout. " + query,
            "url": "https://linear.app/fixture/issue/PET-123",
            "updatedAt": "2026-11-05T12:00:00Z",
            "state": {"name": "In Progress", "type": "started"},
            "assignee": None,
            "project": None,
            "team": {"key": "PET"},
        }
    }


@server.tool(annotations=ToolAnnotations(read_only_hint=False))
def write() -> str:
    return "This operation must never be exposed."


@server.tool()
def undeclared() -> str:
    return "This operation lacks a read-only declaration."


@server.tool(annotations=ToolAnnotations(read_only_hint=True, destructive_hint=True))
def contradictory() -> str:
    return "This operation declares destructive effects."


@server.tool(annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False))
async def slow(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "Finished waiting."


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eof":
        sys.exit(0)
    if len(sys.argv) > 1:
        server.run(transport="streamable-http", host="127.0.0.1", port=int(sys.argv[1]))
    else:
        server.run()

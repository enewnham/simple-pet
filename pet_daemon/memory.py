import json
import re
import sqlite3
import time
from pathlib import Path

from .activity import activity, result_summary
from .prompts import select_prompt
from .telemetry import trace


SCHEMA_VERSION = 5


def bounded(value, limit=240):
    return " ".join(str(value or "").split())[:limit]


class Memory:
    def __init__(self, path, pet, clock=time.time, retention_days=30, max_events=10000):
        self.pet = pet
        self.clock = clock
        self.retention_days = retention_days
        self.max_events = max_events
        path = Path(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (4, SCHEMA_VERSION):
            self.db.executescript("""
                DROP TRIGGER IF EXISTS events_insert;
                DROP TRIGGER IF EXISTS events_delete;
                DROP TABLE IF EXISTS events_fts;
                DROP TABLE IF EXISTS events;
                DROP TABLE IF EXISTS projects;
                DROP TABLE IF EXISTS speech;
                DROP TABLE IF EXISTS contexts;
            """)
        self.db.executescript("""
            PRAGMA auto_vacuum = INCREMENTAL;
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, session TEXT NOT NULL,
                repo TEXT NOT NULL, event_type TEXT NOT NULL, summary TEXT NOT NULL,
                goal TEXT NOT NULL, timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS events_scope ON events(pet, repo, id);
            CREATE INDEX IF NOT EXISTS events_age ON events(timestamp);
            CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(goal, content=events, content_rowid=id);
            CREATE TRIGGER IF NOT EXISTS events_insert AFTER INSERT ON events BEGIN
                INSERT INTO events_fts(rowid, goal) VALUES (new.id, new.goal);
            END;
            CREATE TRIGGER IF NOT EXISTS events_delete AFTER DELETE ON events BEGIN
                INSERT INTO events_fts(events_fts, rowid, goal) VALUES ('delete', old.id, old.goal);
            END;
            CREATE TABLE IF NOT EXISTS projects (
                pet TEXT NOT NULL, repo TEXT NOT NULL, goal TEXT NOT NULL,
                last_result TEXT NOT NULL, timestamp REAL NOT NULL,
                PRIMARY KEY(pet, repo)
            );
            CREATE TABLE IF NOT EXISTS speech (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, text TEXT NOT NULL,
                reason TEXT NOT NULL, timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS contexts (
                pet TEXT NOT NULL, session TEXT NOT NULL, first_prompt TEXT NOT NULL,
                session_context TEXT NOT NULL, recent_requests TEXT NOT NULL, timestamp REAL NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0, first_summary_attempted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(pet, session)
            );
            CREATE TABLE IF NOT EXISTS repository_associations (
                pet TEXT NOT NULL, session TEXT NOT NULL, worktree TEXT NOT NULL,
                PRIMARY KEY(pet, session, worktree)
            );
            CREATE TABLE IF NOT EXISTS repositories (
                pet TEXT NOT NULL, worktree TEXT NOT NULL, observation TEXT NOT NULL,
                PRIMARY KEY(pet, worktree)
            );
        """)
        self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.prune()

    def prune(self):
        cutoff = self.clock() - self.retention_days * 86400
        with self.db:
            self.db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            self.db.execute("DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id DESC LIMIT -1 OFFSET ?)", (self.max_events,))
            self.db.execute("DELETE FROM speech WHERE timestamp < ? OR id NOT IN (SELECT id FROM speech ORDER BY id DESC LIMIT 256)", (cutoff,))
            self.db.execute("DELETE FROM projects WHERE timestamp < ? OR rowid NOT IN (SELECT rowid FROM projects ORDER BY timestamp DESC LIMIT 128)", (cutoff,))
            self.db.execute("DELETE FROM contexts WHERE timestamp < ? OR rowid NOT IN (SELECT rowid FROM contexts ORDER BY timestamp DESC LIMIT 128)", (cutoff,))
        self.db.execute("PRAGMA incremental_vacuum(64)")

    def record_event(self, event, session=None):
        kind = event["event"]
        state = session or {}
        repo = bounded(event.get("repo") or state.get("repo"), 1024)
        identifier = bounded(state.get("id") or f"{event['harness']}:{event['session']}", 256)
        progress = bounded(event.get("progress"))
        goal = bounded(event.get("goal") or state.get("goal"))
        tool = bounded(event.get("tool") or state.get("last_tool"), 160)
        if kind == "tool_start":
            return None
        name = event.get("tool_name", "").lower().rsplit("__", 1)[-1].rsplit(".", 1)[-1]
        read_only = name in ("read", "read_file", "readfile", "grep", "find", "search", "glob")
        if kind == "tool_done" and read_only:
            return None
        if kind == "progress" and not progress:
            return None
        terminal = kind in ("turn_done", "turn_error", "subagent_done", "interrupt", "session_end")
        outcome = state.get("outcome") if terminal and state.get("outcome") else event.get("outcome")
        if kind in ("tool_error", "turn_error"):
            outcome = "error"
        category = state.get("activity") or activity(tool, event.get("tool_name", ""))
        if kind == "turn_error":
            category = "harness"
        if outcome == "error":
            progress = ""
        facts = [kind, goal if kind in ("intent", "subagent_start") else "",
                 result_summary(category, outcome) if outcome else "",
                 tool if kind.startswith("tool_") and outcome != "error" else "", progress]
        summary = bounded(" | ".join(part for part in facts if part), 320)
        if kind in ("progress", "waiting", "session_start"):
            previous = self.db.execute("SELECT event_type,summary FROM events WHERE pet=? AND session=? AND repo=? ORDER BY id DESC LIMIT 1",
                                       (self.pet, identifier, repo)).fetchone()
            if previous and previous["event_type"] == kind and previous["summary"] == summary:
                return None
        timestamp = self.clock()
        with self.db:
            cursor = self.db.execute("INSERT INTO events(pet,session,repo,event_type,summary,goal,timestamp) VALUES (?,?,?,?,?,?,?)",
                                     (self.pet, identifier, repo, kind, summary, goal, timestamp))
            if repo and goal:
                self.db.execute("""INSERT INTO projects VALUES (?,?,?,?,?) ON CONFLICT(pet,repo) DO UPDATE SET
                    goal=excluded.goal, last_result=excluded.last_result, timestamp=excluded.timestamp""",
                    (self.pet, repo, goal, summary, timestamp))
        self.prune()
        trace("memory.event", id=cursor.lastrowid, event_type=kind)
        return cursor.lastrowid

    def recent(self, limit=5):
        rows = self.db.execute("""WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY session ORDER BY id DESC) AS rank FROM events WHERE pet=?
        ) SELECT id, session, repo, event_type, summary, goal, timestamp
          FROM ranked ORDER BY rank=1 DESC, id DESC LIMIT ?""", (self.pet, limit)).fetchall()
        return [dict(row) for row in sorted(rows, key=lambda row: row["id"])]

    def recall(self, event, limit=2):
        if not event or event.get("event") != "intent" or not event.get("repo") or not event.get("goal"):
            return []
        repo = bounded(event["repo"], 1024)
        tokens = re.findall(r"[\w]+", event["goal"])
        distinctive = list(dict.fromkeys(token for token in tokens if len(token) >= 5 and token.lower() not in
                                        {"please", "continue", "failed", "error", "build", "tests", "running"}))[:3]
        if not distinctive:
            return []
        query = " AND ".join('"' + token + '"' for token in distinctive)
        excluded = [row["id"] for row in self.recent()]
        placeholders = ",".join("?" for _ in excluded) or "NULL"
        session = bounded(event.get("target_session") or f"{event.get('harness', '')}:{event.get('session', '')}", 256)
        rows = self.db.execute(f"""SELECT e.* FROM events_fts JOIN events e ON e.id=events_fts.rowid
            WHERE events_fts MATCH ? AND e.pet=? AND e.repo=? AND e.id NOT IN ({placeholders})
            ORDER BY e.session=? DESC, bm25(events_fts), e.id DESC LIMIT ?""",
            (query, self.pet, repo, *excluded, session, limit)).fetchall()
        return [dict(row) for row in rows]

    def projects(self, repos):
        result = []
        for repo in list(dict.fromkeys(repos))[:3]:
            row = self.db.execute("SELECT repo,goal,last_result,timestamp FROM projects WHERE pet=? AND repo=?", (self.pet, repo)).fetchone()
            if row:
                result.append(dict(row))
        return result

    def context(self, session):
        row = self.db.execute("SELECT first_prompt,session_context,recent_requests,request_count,first_summary_attempted FROM contexts WHERE pet=? AND session=?", (self.pet, session)).fetchone()
        if row is None:
            return None
        return {"first_prompt": row["first_prompt"], "session_context": row["session_context"],
                "recent_requests": json.loads(row["recent_requests"]), "request_count": row["request_count"],
                "first_summary_attempted": bool(row["first_summary_attempted"])}

    def save_context(self, session, first_prompt, summary, recent_requests, request_count=None, first_summary_attempted=False):
        requests = [select_prompt(request) for request in recent_requests[-4:]]
        count = len(requests) if request_count is None else request_count
        with self.db:
            self.db.execute("""INSERT INTO contexts VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(pet,session) DO UPDATE SET
                first_prompt=excluded.first_prompt, session_context=excluded.session_context,
                recent_requests=excluded.recent_requests, timestamp=excluded.timestamp,
                request_count=excluded.request_count, first_summary_attempted=excluded.first_summary_attempted""",
                (self.pet, session, select_prompt(first_prompt), bounded(summary, 400), json.dumps(requests), self.clock(),
                 count, first_summary_attempted))
        self.prune()

    def record_speech(self, text, reason):
        with self.db:
            self.db.execute("INSERT INTO speech(pet,text,reason,timestamp) VALUES (?,?,?,?)", (self.pet, text, reason, self.clock()))
        self.prune()

    def associate_repository(self, session, worktree):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO repository_associations VALUES (?,?,?)", (self.pet, session, worktree))

    def repository_associations(self):
        rows = self.db.execute("SELECT session,worktree FROM repository_associations WHERE pet=?", (self.pet,)).fetchall()
        return [dict(row) for row in rows]

    def save_repository(self, worktree, observation):
        with self.db:
            self.db.execute("INSERT INTO repositories VALUES (?,?,?) ON CONFLICT(pet,worktree) DO UPDATE SET observation=excluded.observation",
                            (self.pet, worktree, json.dumps(observation)))

    def load_repositories(self):
        rows = self.db.execute("SELECT observation FROM repositories WHERE pet=?", (self.pet,)).fetchall()
        return [json.loads(row["observation"]) for row in rows]

    def speech(self, limit=3):
        rows = self.db.execute("SELECT text,reason,timestamp FROM speech WHERE pet=? ORDER BY id DESC LIMIT ?", (self.pet, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def clear(self):
        with self.db:
            for table in ("events", "projects", "speech", "contexts", "repository_associations", "repositories"):
                self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))

    def close(self):
        self.db.close()

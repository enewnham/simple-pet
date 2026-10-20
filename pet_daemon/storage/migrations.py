import sqlite3

from .transactions import transaction

SCHEMA_VERSION = 8


MEMORY_SCHEMA = """
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
"""

WORK_SCHEMA = """
            CREATE TABLE IF NOT EXISTS work_items (
                pet TEXT NOT NULL, id TEXT NOT NULL, intent TEXT NOT NULL, status TEXT NOT NULL,
                changed_at REAL NOT NULL, PRIMARY KEY(pet,id));
            CREATE TABLE IF NOT EXISTS work_sessions (
                pet TEXT NOT NULL, session TEXT NOT NULL, work_id TEXT NOT NULL,
                PRIMARY KEY(pet,session));
            CREATE TABLE IF NOT EXISTS work_memberships (
                pet TEXT NOT NULL, work_id TEXT NOT NULL, session TEXT NOT NULL,
                PRIMARY KEY(pet,work_id,session));
            CREATE TABLE IF NOT EXISTS work_aliases (
                pet TEXT NOT NULL, alias TEXT NOT NULL, session TEXT NOT NULL,
                PRIMARY KEY(pet,alias));
            CREATE TABLE IF NOT EXISTS work_artifacts (
                pet TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT NOT NULL,
                repository TEXT NOT NULL, data TEXT NOT NULL, version INTEGER NOT NULL,
                PRIMARY KEY(pet,id));
            CREATE TABLE IF NOT EXISTS work_links (
                pet TEXT NOT NULL, work_id TEXT NOT NULL, artifact_id TEXT NOT NULL,
                PRIMARY KEY(pet,work_id,artifact_id));
            CREATE TABLE IF NOT EXISTS work_evidence (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, work_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL, source TEXT NOT NULL, kind TEXT NOT NULL,
                data TEXT NOT NULL, timestamp REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS work_evidence_scope ON work_evidence(pet,work_id,artifact_id,source,kind,id);
            CREATE TABLE IF NOT EXISTS work_opportunities (
                pet TEXT NOT NULL, id TEXT NOT NULL, work_id TEXT NOT NULL, kind TEXT NOT NULL,
                artifact_id TEXT NOT NULL, version INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                data TEXT NOT NULL, active INTEGER NOT NULL, surfaced_version INTEGER NOT NULL,
                changed_at REAL NOT NULL, PRIMARY KEY(pet,id));
            CREATE TABLE IF NOT EXISTS work_surfacings (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, opportunity_id TEXT NOT NULL,
                version INTEGER NOT NULL, outcome TEXT NOT NULL, timestamp REAL NOT NULL);
"""

DELIVERY_SCHEMA = """
            CREATE TABLE IF NOT EXISTS attention_history (
                pet TEXT NOT NULL, subject TEXT NOT NULL, last_selected REAL NOT NULL,
                considered_revision TEXT NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY(pet,subject));
            CREATE TABLE IF NOT EXISTS message_outcomes (
                pet TEXT NOT NULL, message_id TEXT NOT NULL, outcome TEXT NOT NULL,
                text TEXT NOT NULL, reason TEXT NOT NULL, opportunity_id TEXT,
                version INTEGER, timestamp REAL NOT NULL, PRIMARY KEY(pet,message_id));
"""


def execute_schema(db: sqlite3.Connection, script: str) -> None:
    statement = ""
    for line in script.splitlines():
        statement += line + "\n"
        if sqlite3.complete_statement(statement):
            db.execute(statement)
            statement = ""


DROP_SCHEMA = """
    DROP TABLE IF EXISTS events;
    DROP TABLE IF EXISTS events_fts;
    DROP TABLE IF EXISTS projects;
    DROP TABLE IF EXISTS speech;
    DROP TABLE IF EXISTS contexts;
    DROP TABLE IF EXISTS repository_associations;
    DROP TABLE IF EXISTS repositories;
    DROP TABLE IF EXISTS work_items;
    DROP TABLE IF EXISTS work_sessions;
    DROP TABLE IF EXISTS work_memberships;
    DROP TABLE IF EXISTS work_aliases;
    DROP TABLE IF EXISTS work_artifacts;
    DROP TABLE IF EXISTS work_links;
    DROP TABLE IF EXISTS work_evidence;
    DROP TABLE IF EXISTS work_opportunities;
    DROP TABLE IF EXISTS work_surfacings;
    DROP TABLE IF EXISTS attention_history;
    DROP TABLE IF EXISTS message_outcomes;
    DROP TABLE IF EXISTS work_updates;
    DROP TABLE IF EXISTS work_decisions;
"""


def migrate(db: sqlite3.Connection) -> None:
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    # Rebuild disposable local memory on version changes rather than upgrading saved rows.
    with transaction(db):
        execute_schema(db, DROP_SCHEMA)
        execute_schema(db, MEMORY_SCHEMA)
        execute_schema(db, WORK_SCHEMA)
        execute_schema(db, DELIVERY_SCHEMA)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

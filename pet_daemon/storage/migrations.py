import sqlite3

from .transactions import transaction

SCHEMA_VERSION = 22


MEMORY_SCHEMA = """
            PRAGMA auto_vacuum = INCREMENTAL;
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, source TEXT NOT NULL, session TEXT NOT NULL,
                working_directory TEXT NOT NULL, repository_kind TEXT, repository_value TEXT,
                event_type TEXT NOT NULL, details TEXT NOT NULL,
                goal TEXT NOT NULL, timestamp REAL NOT NULL, data TEXT
            );
            CREATE INDEX IF NOT EXISTS events_scope ON events(pet, working_directory, id);
            CREATE INDEX IF NOT EXISTS events_repository ON events(pet, repository_kind, repository_value, id);
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
                last_request TEXT NOT NULL, timestamp REAL NOT NULL,
                PRIMARY KEY(pet, repo)
            );
            CREATE TABLE IF NOT EXISTS speech (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, text TEXT NOT NULL,
                reason TEXT NOT NULL, timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS contexts (
                pet TEXT NOT NULL, source TEXT NOT NULL, session TEXT NOT NULL, first_prompt TEXT NOT NULL,
                session_context TEXT NOT NULL, recent_requests TEXT NOT NULL, timestamp REAL NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0, first_summary_attempted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(pet, source, session)
            );

"""

WORK_SCHEMA = """
    CREATE TABLE IF NOT EXISTS work_sessions (
        pet TEXT NOT NULL, source TEXT NOT NULL, session TEXT NOT NULL, parent_source TEXT, parent_session TEXT,
        location TEXT NOT NULL DEFAULT '', location_at REAL NOT NULL DEFAULT 0,
        PRIMARY KEY(pet,source,session));
    CREATE TABLE IF NOT EXISTS work_reports (
        id INTEGER PRIMARY KEY, pet TEXT NOT NULL, source TEXT NOT NULL, session TEXT NOT NULL,
        kind TEXT NOT NULL, data TEXT NOT NULL,
        observed_at REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS reports_session ON work_reports(pet,source,session,id);
    CREATE TABLE IF NOT EXISTS work_aliases (
        pet TEXT NOT NULL, alias_source TEXT NOT NULL, alias_session TEXT NOT NULL, source TEXT NOT NULL, session TEXT NOT NULL,
        PRIMARY KEY(pet,alias_source,alias_session));
    CREATE TABLE IF NOT EXISTS work_checkouts (
        pet TEXT NOT NULL, worktree TEXT NOT NULL, data TEXT NOT NULL,
        PRIMARY KEY(pet,worktree));
    CREATE TABLE IF NOT EXISTS work_branches (
        pet TEXT NOT NULL, repository_kind TEXT NOT NULL, repository_value TEXT NOT NULL, branch TEXT NOT NULL, data TEXT NOT NULL,
        PRIMARY KEY(pet,repository_kind,repository_value,branch));
    CREATE TABLE IF NOT EXISTS work_pull_requests (
        pet TEXT NOT NULL, ref TEXT NOT NULL, data TEXT NOT NULL,
        PRIMARY KEY(pet,ref));
"""

DELIVERY_SCHEMA = """
            CREATE TABLE IF NOT EXISTS attention_history (
                pet TEXT NOT NULL, subject_source TEXT NOT NULL, subject_session TEXT NOT NULL, last_selected REAL NOT NULL,
                considered_subject_revision TEXT, considered_session_revision TEXT,
                updated_at REAL NOT NULL, PRIMARY KEY(pet,subject_source,subject_session));
            CREATE TABLE IF NOT EXISTS message_outcomes (
                pet TEXT NOT NULL, message_id TEXT NOT NULL, outcome TEXT NOT NULL,
                text TEXT NOT NULL, reason TEXT NOT NULL, timestamp REAL NOT NULL,
                subject_source TEXT, subject_session TEXT, revision TEXT,
                PRIMARY KEY(pet,message_id));
"""

NARRATIVE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS work_narratives (
        pet TEXT NOT NULL, subject_source TEXT NOT NULL, subject_session TEXT NOT NULL, data TEXT NOT NULL,
        updated_at REAL NOT NULL, PRIMARY KEY(pet,subject_source,subject_session));
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
    DROP TABLE IF EXISTS work_reports;
    DROP TABLE IF EXISTS work_memberships;
    DROP TABLE IF EXISTS work_aliases;
    DROP TABLE IF EXISTS work_artifacts;
    DROP TABLE IF EXISTS work_links;
    DROP TABLE IF EXISTS work_evidence;
    DROP TABLE IF EXISTS work_attempts;
    DROP TABLE IF EXISTS work_checkouts;
    DROP TABLE IF EXISTS work_branches;
    DROP TABLE IF EXISTS work_pull_requests;
    DROP TABLE IF EXISTS work_opportunities;
    DROP TABLE IF EXISTS work_surfacings;
    DROP TABLE IF EXISTS attention_history;
    DROP TABLE IF EXISTS message_outcomes;
    DROP TABLE IF EXISTS work_updates;
    DROP TABLE IF EXISTS work_decisions;
    DROP TABLE IF EXISTS work_narratives;
"""


def migrate(db: sqlite3.Connection) -> None:
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise ValueError(f"Unsupported memory schema version {version}")
    if version == SCHEMA_VERSION:
        return
    with transaction(db):
        execute_schema(db, DROP_SCHEMA)
        execute_schema(db, MEMORY_SCHEMA)
        execute_schema(db, WORK_SCHEMA)
        execute_schema(db, DELIVERY_SCHEMA)
        execute_schema(db, NARRATIVE_SCHEMA)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

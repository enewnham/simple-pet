import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def transaction(db: sqlite3.Connection) -> Generator[None]:
    name = "pet_" + uuid.uuid4().hex
    db.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        db.execute(f"ROLLBACK TO {name}")
        db.execute(f"RELEASE {name}")
        raise
    else:
        db.execute(f"RELEASE {name}")

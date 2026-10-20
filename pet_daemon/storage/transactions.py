from contextlib import contextmanager
import uuid


@contextmanager
def transaction(db):
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

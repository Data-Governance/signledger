"""SEC-SWEEP-2 item 24: cross-process atomic append for the PostgreSQL backend.

Unit tests drive the backend against fake connections (no database needed) and
prove the lock call, the transaction boundary, the retry path and the genesis
rule. test_two_appenders_real_postgres runs only when SIGNLEDGER_TEST_PG_DSN
points at a disposable PostgreSQL database.
"""

import logging
import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import psycopg2
import pytest

from signledger.backends.base import InMemoryBackend
from signledger.backends.postgresql import PostgreSQLBackend
from signledger.core.exceptions import StorageError, ValidationError
from signledger.core.ledger import Entry, Ledger

TABLE = "ledger_entries"
INDEX = f"uq_{TABLE}_previous_hash"


class ChainViolation(psycopg2.IntegrityError):
    diag = SimpleNamespace(constraint_name=INDEX)


class IdViolation(psycopg2.IntegrityError):
    diag = SimpleNamespace(constraint_name="ledger_entries_pkey")

    def __str__(self):
        return "duplicate key value violates unique constraint"


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.events.append(("execute", " ".join(sql.split()), params))
        text_sql = " ".join(sql.split())
        if text_sql.startswith("SELECT hash, seq"):
            self._row = self.conn.tips.pop(0) if self.conn.tips else None
        elif text_sql.startswith("INSERT") and self.conn.insert_errors:
            raise self.conn.insert_errors.pop(0)

    def fetchone(self):
        return self._row


class FakeConn:
    def __init__(self, tips=None, insert_errors=None, events=None):
        self.events = events if events is not None else []
        self.tips = list(tips or [])
        self.insert_errors = list(insert_errors or [])

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.events.append(("commit",))

    def rollback(self):
        self.events.append(("rollback",))


class FakePool:
    def __init__(self, conns):
        self.conns = list(conns)
        self.returned = []

    def getconn(self):
        return self.conns.pop(0)

    def putconn(self, conn):
        self.returned.append(conn)


def make_backend(conns, table=TABLE):
    be = object.__new__(PostgreSQLBackend)
    be.name = "PostgreSQLBackend"
    be.table_name = table
    be.use_sqlalchemy = False
    be._pool = FakePool(conns)
    return be


def make_builder(calls):
    def build(prev):
        calls.append(prev)
        return Entry(data={"n": len(calls)}, previous_hash=prev, hash=f"h{len(calls)}")
    return build


def sql_of(conn):
    return [e[1] for e in conn.events if e[0] == "execute"]


def test_advisory_lock_key_is_stable_int64_and_per_table():
    a = make_backend([], "ledger_a").advisory_lock_key()
    assert a == make_backend([], "ledger_a").advisory_lock_key()
    assert a != make_backend([], "ledger_b").advisory_lock_key()
    assert -(2 ** 63) <= a < 2 ** 63


def test_append_takes_lock_first_and_commits_once_on_one_connection():
    conn = FakeConn(tips=[("tiphash", 7)])
    be = make_backend([conn])
    calls = []
    entry = be.append_chained(make_builder(calls))
    stmts = sql_of(conn)
    assert stmts[0] == "SELECT pg_advisory_xact_lock(%s)"
    assert conn.events[0][2] == (be.advisory_lock_key(),)
    assert stmts[1].startswith("SELECT hash, seq FROM ledger_entries ORDER BY seq DESC")
    assert stmts[2].startswith("INSERT INTO ledger_entries")
    assert [e[0] for e in conn.events].count("commit") == 1
    assert conn.events[-1] == ("commit",)
    assert calls == ["tiphash"]
    assert entry.previous_hash == "tiphash"
    assert be._pool.returned == [conn]


def test_genesis_append_passes_none_as_previous_hash():
    conn = FakeConn(tips=[])
    be = make_backend([conn])
    calls = []
    be.append_chained(make_builder(calls))
    assert calls == [None]


def test_unique_violation_on_previous_hash_retries_once_and_logs_seqs(caplog):
    first = FakeConn(tips=[("old", 7)], insert_errors=[ChainViolation("fork")])
    second = FakeConn(tips=[("newtip", 8)])
    be = make_backend([first, second])
    be.get_latest_entry = lambda: SimpleNamespace(seq_no=8)
    calls = []
    with caplog.at_level(logging.WARNING, logger="signledger.backends.postgresql"):
        entry = be.append_chained(make_builder(calls))
    assert ("rollback",) in first.events
    assert ("commit",) not in first.events
    assert second.events[0][1] == "SELECT pg_advisory_xact_lock(%s)"
    assert second.events[-1] == ("commit",)
    assert calls == ["old", "newtip"]
    assert entry.previous_hash == "newtip"
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "seq was 7" in msgs[0] and "now 8" in msgs[0]


def test_persistent_chain_violation_raises_storage_error_after_one_retry():
    first = FakeConn(tips=[("a", 1)], insert_errors=[ChainViolation("x")])
    second = FakeConn(tips=[("b", 2)], insert_errors=[ChainViolation("x")])
    be = make_backend([first, second])
    be.get_latest_entry = lambda: None
    with pytest.raises(StorageError, match="persisted after retry"):
        be.append_chained(make_builder([]))
    assert be._pool.returned == [first, second]


def test_duplicate_id_is_not_retried():
    conn = FakeConn(tips=[("a", 1)], insert_errors=[IdViolation("dup")])
    be = make_backend([conn])
    with pytest.raises(StorageError, match="already exists"):
        be.append_chained(make_builder([]))
    assert ("rollback",) in conn.events


def test_build_failure_rolls_back_and_propagates_unwrapped():
    conn = FakeConn(tips=[("a", 1)])
    be = make_backend([conn])

    def boom(prev):
        raise ValidationError("Signer function required for signing")

    with pytest.raises(ValidationError):
        be.append_chained(boom)
    assert ("rollback",) in conn.events
    assert ("commit",) not in conn.events


def test_other_database_error_becomes_storage_error():
    class Broken(FakeConn):
        def cursor(self):
            raise psycopg2.OperationalError("connection lost")

    conn = Broken()
    be = make_backend([conn])
    with pytest.raises(StorageError, match="connection lost"):
        be.append_chained(make_builder([]))
    assert ("rollback",) in conn.events


def test_chain_unique_index_ddl_is_idempotent_and_covers_genesis():
    be = make_backend([])
    captured = []
    be._execute_ddl = lambda stmts: captured.extend(stmts)
    be._create_chain_unique_index()
    (stmt,) = captured
    assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_ledger_entries_previous_hash" in stmt
    assert "COALESCE(previous_hash, 'GENESIS')" in stmt


# SQLAlchemy configuration -------------------------------------------------

class FakeSAResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class FakeSAConn:
    def __init__(self, tip, insert_error=None):
        self.statements = []
        self.tip = tip
        self.insert_error = insert_error

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.statements.append(" ".join(sql.split()))
        if sql.lstrip().upper().startswith("INSERT") and self.insert_error:
            raise self.insert_error
        return FakeSAResult(self.tip)

    @contextmanager
    def begin_nested(self):
        yield


class FakeEngine:
    def __init__(self, conns):
        self.conns = list(conns)

    @contextmanager
    def begin(self):
        yield self.conns.pop(0)


def make_sa_backend(conns):
    from sqlalchemy import Column, MetaData, String, Table
    be = object.__new__(PostgreSQLBackend)
    be.name = "PostgreSQLBackend"
    be.table_name = TABLE
    be.use_sqlalchemy = True
    be._engine = FakeEngine(conns)
    md = MetaData()
    be._table = Table(
        TABLE, md,
        Column("id", String, primary_key=True), Column("timestamp", String),
        Column("data", String), Column("hash", String), Column("previous_hash", String),
        Column("signature", String), Column("metadata", String), Column("nonce", String),
    )
    return be


def test_sqlalchemy_path_takes_lock_before_reading_tip():
    conn = FakeSAConn(("tip", 3))
    be = make_sa_backend([conn])
    calls = []
    be.append_chained(make_builder(calls))
    assert conn.statements[0] == "SELECT pg_advisory_xact_lock(:k)"
    assert conn.statements[1].startswith("SELECT hash, seq FROM ledger_entries")
    assert conn.statements[2].startswith("INSERT INTO ledger_entries")
    assert calls == ["tip"]


def test_sqlalchemy_path_retries_once_on_chain_violation():
    from sqlalchemy.exc import IntegrityError as SAIntegrityError
    err = SAIntegrityError("stmt", {}, ChainViolation("fork"))
    first = FakeSAConn(("old", 1), insert_error=err)
    second = FakeSAConn(("new", 2))
    be = make_sa_backend([first, second])
    be.get_latest_entry = lambda: SimpleNamespace(seq_no=2)
    calls = []
    be.append_chained(make_builder(calls))
    assert calls == ["old", "new"]


def test_sqlalchemy_path_other_integrity_error_is_storage_error():
    from sqlalchemy.exc import IntegrityError as SAIntegrityError
    err = SAIntegrityError("stmt", {}, IdViolation("dup"))
    be = make_sa_backend([FakeSAConn(None, insert_error=err)])
    with pytest.raises(StorageError):
        be.append_chained(make_builder([]))


# Ledger and default-backend behaviour -------------------------------------

def test_ledger_append_chains_through_default_backend():
    ledger = Ledger(backend=InMemoryBackend(), auto_verify=False)
    a = ledger.append({"k": 1})
    b = ledger.append({"k": 2})
    assert a.previous_hash is None
    assert b.previous_hash == a.hash


def test_ledger_missing_signer_still_raises_validation_error():
    ledger = Ledger(backend=InMemoryBackend(), auto_verify=False, enable_signatures=True)
    with pytest.raises(ValidationError):
        ledger.append({"k": 1})


def test_ledger_wraps_backend_failure_in_storage_error():
    class Failing(InMemoryBackend):
        def append_entry(self, entry):
            raise RuntimeError("disk gone")

    ledger = Ledger(backend=Failing(), auto_verify=False)
    with pytest.raises(StorageError, match="disk gone"):
        ledger.append({"k": 1})


def test_default_append_chained_tolerates_tip_read_failure(caplog):
    class BadTip(InMemoryBackend):
        def get_latest_entry(self):
            raise RuntimeError("read failed")

    be = BadTip()
    with caplog.at_level(logging.WARNING, logger="signledger.backends.base"):
        entry = be.append_chained(make_builder([]))
    assert entry.previous_hash is None
    assert any("read failed" in r.getMessage() for r in caplog.records)


# Real PostgreSQL (opt-in) --------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("SIGNLEDGER_TEST_PG_DSN"),
    reason="set SIGNLEDGER_TEST_PG_DSN to a disposable PostgreSQL database",
)
def test_two_appenders_real_postgres_keep_a_single_chain():
    dsn = os.environ["SIGNLEDGER_TEST_PG_DSN"]
    table = "sl_atomic_test"
    # Separate backends and Ledgers share no in-process lock, like two tasks.
    ledgers = [
        Ledger(backend=PostgreSQLBackend(dsn, table_name=table), auto_verify=False)
        for _ in range(2)
    ]
    errors = []

    def work(ledger, tag):
        try:
            for i in range(25):
                ledger.append({"w": tag, "i": i})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(lg, n)) for n, lg in enumerate(ledgers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries = list(ledgers[0].backend.get_entries())
    assert len(entries) == 50
    assert len({e.previous_hash for e in entries}) == 50
    assert ledgers[0].verify_integrity().is_valid

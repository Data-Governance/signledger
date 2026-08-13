"""PostgreSQL backend for SignLedger."""

from datetime import datetime
from typing import Optional, Iterator, Dict, Any
import json
import logging

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from psycopg2.pool import SimpleConnectionPool
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, Integer, BigInteger, JSON, Index, Identity, text
    from sqlalchemy.dialects.postgresql import UUID
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from .base import StorageBackend
from ..core.exceptions import StorageError
from ..core.ledger import Entry

logger = logging.getLogger(__name__)


class PostgreSQLBackend(StorageBackend):
    """PostgreSQL storage backend for SignLedger."""
    
    def __init__(
        self,
        connection_string: str,
        table_name: str = "ledger_entries",
        pool_size: int = 5,
        max_overflow: int = 10,
        use_sqlalchemy: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if use_sqlalchemy and not HAS_SQLALCHEMY:
            raise ImportError("SQLAlchemy is required. Install with: pip install signledger[postgresql]")
        elif not use_sqlalchemy and not HAS_PSYCOPG2:
            raise ImportError("psycopg2 is required. Install with: pip install signledger[postgresql]")
        
        self.connection_string = connection_string
        self.table_name = table_name
        self.use_sqlalchemy = use_sqlalchemy
        # Checkpoint table name is derived from the entries table name so a
        # custom table_name still gets a matching checkpoint table.
        self.checkpoint_table_name = f"{table_name}_checkpoints"

        if use_sqlalchemy:
            self._setup_sqlalchemy(pool_size, max_overflow)
        else:
            self._setup_psycopg2(pool_size)

        # Create table if it doesn't exist
        self.create_indexes()
        # Append-only enforcement at the DB level (REVOKE + trigger) and the
        # checkpoint table. Idempotent; safe to run on every startup.
        self._apply_immutability_guards()
        self._create_checkpoint_table()
    
    def _setup_psycopg2(self, pool_size: int):
        """Setup psycopg2 connection pool."""
        self._pool = SimpleConnectionPool(
            1,
            pool_size,
            self.connection_string
        )
    
    def _setup_sqlalchemy(self, pool_size: int, max_overflow: int):
        """Setup SQLAlchemy engine and metadata."""
        self._engine = create_engine(
            self.connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            json_serializer=lambda obj: json.dumps(obj, default=str),
            json_deserializer=json.loads,
        )
        
        self._metadata = MetaData()
        self._table = Table(
            self.table_name,
            self._metadata,
            Column('id', UUID(as_uuid=False), primary_key=True),
            # Monotonic, DB-assigned, strictly-increasing sequence. GENERATED
            # ALWAYS AS IDENTITY -- the application cannot supply or override it,
            # so it is a tamper-evident witness for tail-truncation and the
            # deterministic ordering key (replaces ORDER BY timestamp).
            Column('seq', BigInteger, Identity(always=True), nullable=False, unique=True),
            Column('timestamp', DateTime(timezone=True), nullable=False, index=True),
            Column('data', JSON, nullable=False),
            Column('hash', String(64), nullable=False, unique=True, index=True),
            Column('previous_hash', String(64), index=True),
            Column('signature', String),
            Column('metadata', JSON),
            Column('nonce', Integer, default=0),
            Column('created_at', DateTime(timezone=True), server_default=text('CURRENT_TIMESTAMP')),
            Index('idx_timestamp_desc', 'timestamp', postgresql_using='btree', postgresql_ops={'timestamp': 'DESC'}),
            Index(f'idx_{self.table_name}_seq', 'seq'),
        )
    
    def _get_connection(self):
        """Get database connection."""
        if self.use_sqlalchemy:
            return self._engine.connect()
        else:
            return self._pool.getconn()
    
    def _put_connection(self, conn):
        """Return connection to pool.

        Tolerates an already-closed pool: a streaming get_entries() generator
        that is abandoned mid-iteration (e.g. verify_integrity raising before the
        iterator is exhausted, then the backend being closed) runs its finally
        block here during GC. Returning to a closed pool would otherwise raise a
        spurious PoolError on an unrelated code path.
        """
        if not self.use_sqlalchemy:
            try:
                self._pool.putconn(conn)
            except Exception:
                # Pool already closed (backend shut down) -- close the raw
                # connection directly so it is not leaked.
                try:
                    conn.close()
                except Exception:
                    pass
    
    def create_indexes(self) -> None:
        """Create table and indexes."""
        if self.use_sqlalchemy:
            self._metadata.create_all(self._engine)
        else:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    # Create table. seq is a DB-assigned monotonic, strictly
                    # increasing identity column -- the application cannot supply
                    # or override it (GENERATED ALWAYS). It is the deterministic
                    # ordering key and the witness for tail-truncation detection.
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id UUID PRIMARY KEY,
                            seq BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
                            timestamp TIMESTAMPTZ NOT NULL,
                            data JSONB NOT NULL,
                            hash VARCHAR(64) NOT NULL UNIQUE,
                            previous_hash VARCHAR(64),
                            signature TEXT,
                            metadata JSONB,
                            nonce INTEGER DEFAULT 0,
                            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Idempotent upgrade path for tables created before the seq
                    # column existed: add it (GENERATED ALWAYS AS IDENTITY
                    # backfills existing rows in physical order, then continues
                    # strictly increasing). ADD COLUMN IF NOT EXISTS makes this
                    # safe to run on every startup.
                    cur.execute(f"""
                        ALTER TABLE {self.table_name}
                        ADD COLUMN IF NOT EXISTS seq BIGINT GENERATED ALWAYS AS IDENTITY
                    """)
                    cur.execute(f"""
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_{self.table_name}_seq_unique
                        ON {self.table_name} (seq)
                    """)

                    # Create indexes
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
                        ON {self.table_name} (timestamp DESC)
                    """)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_hash 
                        ON {self.table_name} (hash)
                    """)
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_previous_hash 
                        ON {self.table_name} (previous_hash)
                    """)
                    
                    # Create GIN index for JSONB data
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data_gin 
                        ON {self.table_name} USING GIN (data)
                    """)
                    
                    conn.commit()
            finally:
                self._put_connection(conn)

    def _execute_ddl(self, statements: list) -> None:
        """Run a list of DDL statements in one transaction.

        Works for both the psycopg2 and SQLAlchemy configurations -- the
        immutability guards and checkpoint table are plain SQL with no
        dialect-specific bind parameters, so we route them through a raw DBAPI
        cursor in both cases.
        """
        if self.use_sqlalchemy:
            with self._engine.begin() as conn:
                for stmt in statements:
                    conn.execute(text(stmt))
        else:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    for stmt in statements:
                        cur.execute(stmt)
                conn.commit()
            finally:
                self._put_connection(conn)

    def _apply_immutability_guards(self) -> None:
        """Make ledger_entries append-only AT THE DATABASE, not by convention.

        Two layers, idempotent (safe to run on every startup):

        1. REVOKE UPDATE, DELETE, TRUNCATE from PUBLIC. This enforces append-only
           for any NON-superuser application role -- the least-privilege
           deployment posture.

        2. A BEFORE UPDATE OR DELETE trigger that RAISEs. This is the REAL
           always-on guard: PostgreSQL superusers BYPASS table privileges (and
           RLS), so REVOKE alone does NOT stop a superuser connection. Triggers,
           however, fire for EVERY role including superusers, so the trigger
           blocks UPDATE/DELETE even when the service connects as a superuser
           (which the current deployment does -- only the `postgres` role
           exists). TRUNCATE is additionally covered by a BEFORE TRUNCATE
           statement-level trigger because TRUNCATE does not fire row triggers.

        CREATE OR REPLACE FUNCTION + DROP TRIGGER IF EXISTS then CREATE keeps the
        whole thing re-runnable without error.
        """
        fn_name = f"{self.table_name}_block_mutation"
        row_trigger = f"{self.table_name}_no_update_delete"
        trunc_trigger = f"{self.table_name}_no_truncate"
        statements = [
            # Least-privilege: non-superuser roles cannot mutate or truncate.
            f"REVOKE UPDATE, DELETE, TRUNCATE ON {self.table_name} FROM PUBLIC",
            # Row-level guard for UPDATE/DELETE (fires for superusers too).
            f"""
            CREATE OR REPLACE FUNCTION {fn_name}()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $body$
            BEGIN
                RAISE EXCEPTION
                    'ledger_entries is append-only: % is not permitted on %',
                    TG_OP, TG_TABLE_NAME
                    USING ERRCODE = 'raise_exception';
            END;
            $body$
            """,
            f"DROP TRIGGER IF EXISTS {row_trigger} ON {self.table_name}",
            f"""
            CREATE TRIGGER {row_trigger}
            BEFORE UPDATE OR DELETE ON {self.table_name}
            FOR EACH ROW EXECUTE FUNCTION {fn_name}()
            """,
            # Statement-level guard for TRUNCATE (row triggers do not see it).
            f"DROP TRIGGER IF EXISTS {trunc_trigger} ON {self.table_name}",
            f"""
            CREATE TRIGGER {trunc_trigger}
            BEFORE TRUNCATE ON {self.table_name}
            FOR EACH STATEMENT EXECUTE FUNCTION {fn_name}()
            """,
        ]
        self._execute_ddl(statements)

    def _create_checkpoint_table(self) -> None:
        """Create the append-only checkpoint table.

        A checkpoint is a signed snapshot of (total_entries, max_seq, tip hash,
        timestamp). Persisting it lets verify_integrity detect tail-truncation
        even on a replica that has no seq gap visible to it: a later observed
        total/max_seq that is LOWER than the latest stored checkpoint proves rows
        were removed. The checkpoint table itself is made append-only by the same
        guard pattern.
        """
        ck = self.checkpoint_table_name
        self._execute_ddl([
            f"""
            CREATE TABLE IF NOT EXISTS {ck} (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                total_entries BIGINT NOT NULL,
                max_seq BIGINT,
                tip_hash VARCHAR(64),
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                signature TEXT,
                algorithm VARCHAR(32)
            )
            """,
            f"REVOKE UPDATE, DELETE, TRUNCATE ON {ck} FROM PUBLIC",
        ])

    def save_checkpoint(
        self,
        total_entries: int,
        max_seq: Optional[int],
        tip_hash: Optional[str],
        signature: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> None:
        """Persist a checkpoint row (append-only)."""
        self._run_insert(
            f"""
            INSERT INTO {self.checkpoint_table_name}
                (total_entries, max_seq, tip_hash, signature, algorithm)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (total_entries, max_seq, tip_hash, signature, algorithm),
        )

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Return the most recently stored checkpoint, or None if there is none."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT total_entries, max_seq, tip_hash, created_at,
                           signature, algorithm
                    FROM {self.checkpoint_table_name}
                    ORDER BY id DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._put_connection(conn)

    def _run_insert(self, sql: str, params: tuple) -> None:
        """Execute a parameterised INSERT in its own transaction."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_connection(conn)

    def get_seq_summary(self) -> Dict[str, Any]:
        """Return seq-stream summary used for tail-truncation detection.

        Returns count, min_seq, max_seq. A healthy append-only ledger satisfies
        (max_seq - min_seq + 1) == count AND there are no duplicate seq values
        (the latter is guaranteed by the UNIQUE constraint, so a violation can
        only mean rows were removed -- i.e. a gap -- which this surfaces).
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*), MIN(seq), MAX(seq) FROM {self.table_name}"
                )
                count, min_seq, max_seq = cur.fetchone()
                return {
                    "count": int(count),
                    "min_seq": int(min_seq) if min_seq is not None else None,
                    "max_seq": int(max_seq) if max_seq is not None else None,
                }
        finally:
            self._put_connection(conn)

    def append_entry(self, entry: Entry) -> None:
        """Append entry to the ledger."""
        if self.use_sqlalchemy:
            self._append_entry_sqlalchemy(entry)
        else:
            self._append_entry_psycopg2(entry)
    
    def _append_entry_psycopg2(self, entry: Entry) -> None:
        """Append entry using psycopg2."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_name} 
                    (id, timestamp, data, hash, previous_hash, signature, metadata, nonce)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    entry.id,
                    entry.timestamp,
                    Json(entry.data),
                    entry.hash,
                    entry.previous_hash,
                    entry.signature,
                    Json(entry.metadata) if entry.metadata else None,
                    entry.nonce,
                ))
                conn.commit()
        except psycopg2.IntegrityError as e:
            conn.rollback()
            if 'duplicate key value violates unique constraint' in str(e):
                raise StorageError(f"Entry with ID {entry.id} already exists", "append", self.name)
            raise StorageError(f"Failed to append entry: {e}", "append", self.name)
        except Exception as e:
            conn.rollback()
            raise StorageError(f"Failed to append entry: {e}", "append", self.name)
        finally:
            self._put_connection(conn)
    
    def _append_entry_sqlalchemy(self, entry: Entry) -> None:
        """Append entry using SQLAlchemy."""
        with self._engine.begin() as conn:
            try:
                conn.execute(self._table.insert().values(
                    id=entry.id,
                    timestamp=entry.timestamp,
                    data=entry.data,
                    hash=entry.hash,
                    previous_hash=entry.previous_hash,
                    signature=entry.signature,
                    metadata=entry.metadata,
                    nonce=entry.nonce,
                ))
            except Exception as e:
                raise StorageError(f"Failed to append entry: {e}", "append", self.name)
    
    def get_entry(self, entry_id: str) -> Optional[Entry]:
        """Get entry by ID."""
        if self.use_sqlalchemy:
            return self._get_entry_sqlalchemy(entry_id)
        else:
            return self._get_entry_psycopg2(entry_id)
    
    def _get_entry_psycopg2(self, entry_id: str) -> Optional[Entry]:
        """Get entry using psycopg2."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT * FROM {self.table_name} WHERE id = %s
                """, (entry_id,))
                row = cur.fetchone()
                
                if row:
                    return Entry.from_dict(self._row_to_dict(row))
                return None
        finally:
            self._put_connection(conn)
    
    def _get_entry_sqlalchemy(self, entry_id: str) -> Optional[Entry]:
        """Get entry using SQLAlchemy."""
        with self._engine.connect() as conn:
            result = conn.execute(
                self._table.select().where(self._table.c.id == entry_id)
            ).first()
            
            if result:
                return Entry.from_dict(self._row_to_dict(result))
            return None
    
    def get_entries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[Entry]:
        """Get entries within time range."""
        if self.use_sqlalchemy:
            yield from self._get_entries_sqlalchemy(start_time, end_time, limit, offset)
        else:
            yield from self._get_entries_psycopg2(start_time, end_time, limit, offset)
    
    def _get_entries_psycopg2(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[Entry]:
        """Get entries using psycopg2."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query
                query = f"SELECT * FROM {self.table_name}"
                params = []
                conditions = []
                
                if start_time:
                    conditions.append("timestamp >= %s")
                    params.append(start_time)
                
                if end_time:
                    conditions.append("timestamp <= %s")
                    params.append(end_time)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                # Order by the DB-assigned monotonic seq, NOT timestamp.
                # Two entries can share a timestamp (or clocks can skew across
                # replicas); seq is strictly increasing and append-order-exact,
                # so it is the only deterministic, tamper-evident ordering key.
                query += " ORDER BY seq ASC"
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                if offset:
                    query += " OFFSET %s"
                    params.append(offset)
                
                cur.execute(query, params)
                
                for row in cur:
                    yield Entry.from_dict(self._row_to_dict(row))
        finally:
            self._put_connection(conn)
    
    def _get_entries_sqlalchemy(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[Entry]:
        """Get entries using SQLAlchemy."""
        with self._engine.connect() as conn:
            query = self._table.select()
            
            if start_time:
                query = query.where(self._table.c.timestamp >= start_time)
            
            if end_time:
                query = query.where(self._table.c.timestamp <= end_time)
            
            # Deterministic, tamper-evident ordering by DB-assigned seq.
            query = query.order_by(self._table.c.seq.asc())

            if limit:
                query = query.limit(limit)

            if offset:
                query = query.offset(offset)

            result = conn.execute(query)
            
            for row in result:
                yield Entry.from_dict(self._row_to_dict(row))
    
    def get_latest_entry(self) -> Optional[Entry]:
        """Get the most recent entry."""
        # "Latest" must be the highest seq -- this entry's hash becomes the
        # previous_hash of the next append, so ordering by timestamp (which can
        # tie or skew) would corrupt the chain. Order by seq.
        if self.use_sqlalchemy:
            with self._engine.connect() as conn:
                result = conn.execute(
                    self._table.select()
                    .order_by(self._table.c.seq.desc())
                    .limit(1)
                ).first()

                if result:
                    return Entry.from_dict(self._row_to_dict(result))
        else:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT * FROM {self.table_name}
                        ORDER BY seq DESC LIMIT 1
                    """)
                    row = cur.fetchone()

                    if row:
                        return Entry.from_dict(self._row_to_dict(row))
            finally:
                self._put_connection(conn)

        return None

    def get_oldest_entry(self) -> Optional[Entry]:
        """Get the oldest entry (lowest seq)."""
        if self.use_sqlalchemy:
            with self._engine.connect() as conn:
                result = conn.execute(
                    self._table.select()
                    .order_by(self._table.c.seq.asc())
                    .limit(1)
                ).first()

                if result:
                    return Entry.from_dict(self._row_to_dict(result))
        else:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT * FROM {self.table_name}
                        ORDER BY seq ASC LIMIT 1
                    """)
                    row = cur.fetchone()

                    if row:
                        return Entry.from_dict(self._row_to_dict(row))
            finally:
                self._put_connection(conn)

        return None
    
    def count_entries(self) -> int:
        """Get total number of entries."""
        if self.use_sqlalchemy:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {self.table_name}")
                ).scalar()
                return result or 0
        else:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                    return cur.fetchone()[0]
            finally:
                self._put_connection(conn)
    
    def get_size(self) -> int:
        """Get approximate storage size in bytes."""
        if self.use_sqlalchemy:
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT pg_total_relation_size('{self.table_name}')")
                ).scalar()
                return result or 0
        else:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT pg_total_relation_size('{self.table_name}')")
                    return cur.fetchone()[0]
            finally:
                self._put_connection(conn)
    
    def close(self) -> None:
        """Close database connections."""
        if self.use_sqlalchemy:
            self._engine.dispose()
        else:
            self._pool.closeall()
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        if isinstance(row, dict):
            # psycopg2 RealDictCursor
            result = dict(row)
        else:
            # SQLAlchemy row
            result = dict(row._mapping)
        
        # Remove created_at as it's not part of Entry
        result.pop('created_at', None)

        # Map the DB-assigned monotonic column `seq` onto Entry.seq_no. seq_no
        # is deliberately NOT in the hash input (see Entry.to_dict), so carrying
        # it on the Entry does not affect hash verification.
        if 'seq' in result:
            result['seq_no'] = result.pop('seq')

        # Empty metadata is stored as SQL NULL (see _append_entry_psycopg2:
        # `Json(entry.metadata) if entry.metadata else None`). Entry.metadata is
        # a non-optional dict, so coerce NULL back to {} on read. An entry hashed
        # with metadata={} also hashes the same way here, so this preserves
        # verification.
        if result.get('metadata') is None:
            result['metadata'] = {}

        return result
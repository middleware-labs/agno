"""Migration v2.5.7: Composite PK on sessions table (session_id, session_type)

Fixes a bug where different entity types (agent, team, workflow) sharing the
same session_id would overwrite each other.  After this migration each
(session_id, session_type) pair is a separate row.

Steps (in a single transaction):
1. Fill any NULL session_type values with 'agent'
2. Drop the old single-column PK on session_id
3. Add composite PK on (session_id, session_type)
"""

from agno.db.base import AsyncBaseDb, BaseDb
from agno.db.migrations.utils import quote_db_identifier
from agno.utils.log import log_error, log_info

try:
    from sqlalchemy import text
except ImportError:
    raise ImportError("`sqlalchemy` not installed. Please install it using `pip install sqlalchemy`")


# ---------------------------------------------------------------------------
# Public entry points (called by MigrationManager)
# ---------------------------------------------------------------------------

def up(db: BaseDb, table_type: str, table_name: str) -> bool:
    if table_type != "sessions":
        return False
    db_type = type(db).__name__
    try:
        if db_type == "PostgresDb":
            return _migrate_postgres(db, table_name)
        log_info(f"v2.5.7: {db_type} not supported for this migration, skipping")
        return False
    except Exception as e:
        log_error(f"v2.5.7 migration failed for {db_type} on {table_name}: {e}")
        raise


async def async_up(db: AsyncBaseDb, table_type: str, table_name: str) -> bool:
    if table_type != "sessions":
        return False
    db_type = type(db).__name__
    try:
        if db_type == "AsyncPostgresDb":
            return await _async_migrate_postgres(db, table_name)
        log_info(f"v2.5.7: {db_type} not supported for this migration, skipping")
        return False
    except Exception as e:
        log_error(f"v2.5.7 async migration failed for {db_type} on {table_name}: {e}")
        raise


def down(db: BaseDb, table_type: str, table_name: str) -> bool:
    if table_type != "sessions":
        return False
    db_type = type(db).__name__
    try:
        if db_type == "PostgresDb":
            return _revert_postgres(db, table_name)
        return False
    except Exception as e:
        log_error(f"v2.5.7 revert failed for {db_type} on {table_name}: {e}")
        raise


async def async_down(db: AsyncBaseDb, table_type: str, table_name: str) -> bool:
    if table_type != "sessions":
        return False
    db_type = type(db).__name__
    try:
        if db_type == "AsyncPostgresDb":
            return await _async_revert_postgres(db, table_name)
        return False
    except Exception as e:
        log_error(f"v2.5.7 async revert failed for {db_type} on {table_name}: {e}")
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pk_has_session_type(sess, db_schema: str, table_name: str) -> bool:
    """Check if the PK already includes session_type."""
    result = sess.execute(
        text(
            "SELECT a.attname "
            "FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "JOIN pg_class c ON c.oid = i.indrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relname = :table AND i.indisprimary"
        ),
        {"schema": db_schema, "table": table_name},
    ).fetchall()
    return "session_type" in [r[0] for r in result]


async def _async_pk_has_session_type(sess, db_schema: str, table_name: str) -> bool:
    result = await sess.execute(
        text(
            "SELECT a.attname "
            "FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "JOIN pg_class c ON c.oid = i.indrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = :schema AND c.relname = :table AND i.indisprimary"
        ),
        {"schema": db_schema, "table": table_name},
    )
    return "session_type" in [r[0] for r in result.fetchall()]


# ---------------------------------------------------------------------------
# PostgreSQL sync
# ---------------------------------------------------------------------------

def _migrate_postgres(db: BaseDb, table_name: str) -> bool:
    db_schema = db.db_schema or "public"  # type: ignore
    db_type = type(db).__name__
    qs = quote_db_identifier(db_type, db_schema)
    qt = quote_db_identifier(db_type, table_name)
    full = f"{qs}.{qt}"
    pkey = quote_db_identifier(db_type, f"{table_name}_pkey")

    with db.Session() as sess, sess.begin():  # type: ignore
        if not sess.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = :s AND table_name = :t)"),
            {"s": db_schema, "t": table_name},
        ).scalar():
            log_info(f"v2.5.7: table {table_name} does not exist, skipping")
            return False

        if _pk_has_session_type(sess, db_schema, table_name):
            log_info(f"v2.5.7: {table_name} already has composite PK, skipping")
            return False

        log_info(f"v2.5.7: migrating {table_name} PK → (session_id, session_type)")

        sess.execute(text(f"UPDATE {full} SET session_type = 'agent' WHERE session_type IS NULL"))
        sess.execute(text(f"ALTER TABLE {full} DROP CONSTRAINT IF EXISTS {pkey}"))
        sess.execute(text(f"ALTER TABLE {full} ADD PRIMARY KEY (session_id, session_type)"))

        log_info(f"v2.5.7: {table_name} migration complete")
        return True


# ---------------------------------------------------------------------------
# PostgreSQL async
# ---------------------------------------------------------------------------

async def _async_migrate_postgres(db: AsyncBaseDb, table_name: str) -> bool:
    db_schema = db.db_schema or "public"  # type: ignore
    db_type = type(db).__name__
    qs = quote_db_identifier(db_type, db_schema)
    qt = quote_db_identifier(db_type, table_name)
    full = f"{qs}.{qt}"
    pkey = quote_db_identifier(db_type, f"{table_name}_pkey")

    async with db.async_session_factory() as sess:  # type: ignore
        async with sess.begin():
            if not (await sess.execute(
                text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = :s AND table_name = :t)"),
                {"s": db_schema, "t": table_name},
            )).scalar():
                log_info(f"v2.5.7: table {table_name} does not exist, skipping")
                return False

            if await _async_pk_has_session_type(sess, db_schema, table_name):
                log_info(f"v2.5.7: {table_name} already has composite PK, skipping")
                return False

            log_info(f"v2.5.7: migrating {table_name} PK → (session_id, session_type)")

            await sess.execute(text(f"UPDATE {full} SET session_type = 'agent' WHERE session_type IS NULL"))
            await sess.execute(text(f"ALTER TABLE {full} DROP CONSTRAINT IF EXISTS {pkey}"))
            await sess.execute(text(f"ALTER TABLE {full} ADD PRIMARY KEY (session_id, session_type)"))

            log_info(f"v2.5.7: {table_name} migration complete")
            return True


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------

def _revert_postgres(db: BaseDb, table_name: str) -> bool:
    db_schema = db.db_schema or "public"  # type: ignore
    db_type = type(db).__name__
    qs = quote_db_identifier(db_type, db_schema)
    qt = quote_db_identifier(db_type, table_name)
    full = f"{qs}.{qt}"
    pkey = quote_db_identifier(db_type, f"{table_name}_pkey")

    with db.Session() as sess, sess.begin():  # type: ignore
        sess.execute(text(f"ALTER TABLE {full} DROP CONSTRAINT IF EXISTS {pkey}"))
        sess.execute(text(f"ALTER TABLE {full} ADD PRIMARY KEY (session_id)"))
        log_info(f"v2.5.7: reverted {table_name} PK to session_id only")
        return True


async def _async_revert_postgres(db: AsyncBaseDb, table_name: str) -> bool:
    db_schema = db.db_schema or "public"  # type: ignore
    db_type = type(db).__name__
    qs = quote_db_identifier(db_type, db_schema)
    qt = quote_db_identifier(db_type, table_name)
    full = f"{qs}.{qt}"
    pkey = quote_db_identifier(db_type, f"{table_name}_pkey")

    async with db.async_session_factory() as sess:  # type: ignore
        async with sess.begin():
            await sess.execute(text(f"ALTER TABLE {full} DROP CONSTRAINT IF EXISTS {pkey}"))
            await sess.execute(text(f"ALTER TABLE {full} ADD PRIMARY KEY (session_id)"))
            log_info(f"v2.5.7: reverted {table_name} PK to session_id only")
            return True

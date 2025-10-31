import pytest

from modules.config.settings import settings
from modules.database.models import DatabaseManager


class FakeCursor:
    def __init__(self, executed):
        self._executed = executed
        self.lastrowid = 1

    def execute(self, query, params=None):
        normalized = " ".join(str(query).split())
        self._executed.append(normalized)

    def fetchone(self):
        return (0,)

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self, executed):
        self._executed = executed

    def cursor(self):
        return FakeCursor(self._executed)

    def commit(self):
        pass

    def close(self):
        pass

    def rollback(self):
        pass


@pytest.mark.parametrize("db_flag", [(True, True), (False, False)])
def test_init_database_creates_subscription_tables(monkeypatch, db_flag):
    use_rds, is_postgres = db_flag
    manager = DatabaseManager()
    manager.use_rds = use_rds
    manager.is_postgres = is_postgres

    executed = []

    def fake_get_connection():
        return FakeConnection(executed)

    monkeypatch.setattr(manager, "get_connection", fake_get_connection)
    monkeypatch.setattr(manager, "ensure_pgvector_extension", lambda: True)
    monkeypatch.setattr(manager, "create_new_tables", lambda: None)
    monkeypatch.setattr(manager, "migrate_documents_table", lambda: None)
    monkeypatch.setattr(manager, "_migrate_documents_schema", lambda: None)
    monkeypatch.setattr(manager, "_migrate_email_verification_schema", lambda: None)
    monkeypatch.setattr(manager, "_migrate_user_profile_schema", lambda: None)
    monkeypatch.setattr(manager, "_migrate_session_schema", lambda: None)

    manager.init_database()

    if use_rds and is_postgres:
        assert any("CREATE TABLE IF NOT EXISTS admin_users" in q for q in executed)
        assert any("CREATE TABLE IF NOT EXISTS subscription_plans" in q for q in executed)
        assert any("CREATE TABLE IF NOT EXISTS subscription_plan_prices" in q for q in executed)
        assert any("CREATE TABLE IF NOT EXISTS user_subscriptions" in q for q in executed)
        assert any("CREATE INDEX IF NOT EXISTS idx_plan_prices_plan" in q for q in executed)
    else:
        assert executed  # sanity check the fake cursor was exercised


def test_get_all_subscription_plans_recovers_missing_price_table(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "USE_RDS", False, raising=False)
    monkeypatch.setattr(settings, "DB_PORT", 0, raising=False)
    test_db = tmp_path / "missing_prices.sqlite"
    manager = DatabaseManager(db_name=str(test_db))
    manager.use_rds = False
    manager.is_postgres = False

    # Simulate the table being dropped in an existing deployment
    conn = manager.get_connection()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS subscription_plan_prices")
    conn.commit()
    conn.close()

    # The new guard should silently recreate the table and not raise
    plans = manager.get_all_subscription_plans()
    assert isinstance(plans, list)

    # Verify the table now exists again
    conn = manager.get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='subscription_plan_prices'"
    )
    assert cur.fetchone()
    conn.close()

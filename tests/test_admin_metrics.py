import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from modules.admin.service import admin_service


def test_get_dashboard_metrics_uses_database(monkeypatch):
    expected = {
        "labels": ["2025-01-01"],
        "series": {"new_users": [1]},
        "totals": {"new_users": 1},
        "range": {"start": "2025-01-01", "end": "2025-01-01"},
    }

    class StubDB:
        def get_dashboard_metrics(self, days):
            assert days == 7
            return expected

    monkeypatch.setattr(admin_service, "db", StubDB())

    result = admin_service.get_dashboard_metrics(7)
    assert result == expected

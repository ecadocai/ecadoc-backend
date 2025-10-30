import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test-key")

dummy_agent = types.ModuleType("modules.agent")
workflow_module = types.ModuleType("modules.agent.workflow")
workflow_module.agent_workflow = None
dummy_agent.workflow = workflow_module
sys.modules.setdefault("modules.agent", dummy_agent)
sys.modules.setdefault("modules.agent.workflow", workflow_module)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.projects.endpoints import router
from modules.projects import service as project_service_module
from modules.auth.deps import get_current_user_id


def create_app():
    app = FastAPI()
    app.include_router(router)
    return app


def test_accept_invitation_endpoint(monkeypatch):
    app = create_app()

    app.dependency_overrides[get_current_user_id] = lambda: 42

    captured = {}

    def fake_respond(invitation_id, user_id, accept):
        captured.update({"invitation_id": invitation_id, "user_id": user_id, "accept": accept})
        return {"status": "accepted"}

    monkeypatch.setattr(project_service_module.project_service, "respond_to_invitation", fake_respond)

    client = TestClient(app)
    response = client.post("/projects/invitations/5/accept")

    assert response.status_code == 200
    assert captured == {"invitation_id": 5, "user_id": 42, "accept": True}


def test_reject_invitation_endpoint(monkeypatch):
    app = create_app()
    app.dependency_overrides[get_current_user_id] = lambda: 77

    captured = {}

    def fake_respond(invitation_id, user_id, accept):
        captured.update({"invitation_id": invitation_id, "user_id": user_id, "accept": accept})
        return {"status": "rejected"}

    monkeypatch.setattr(project_service_module.project_service, "respond_to_invitation", fake_respond)

    client = TestClient(app)
    response = client.post("/projects/invitations/9/reject")

    assert response.status_code == 200
    assert captured == {"invitation_id": 9, "user_id": 77, "accept": False}

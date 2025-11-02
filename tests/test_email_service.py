import pytest

from modules.auth.email_service import email_service


def test_project_invitation_email_includes_view_button(monkeypatch):
    monkeypatch.setattr(email_service, "is_configured", lambda: True)

    captured = {}

    def fake_send(to_email, subject, text_body, html_body):
        captured.update(
            {
                "to_email": to_email,
                "subject": subject,
                "text": text_body,
                "html": html_body,
            }
        )
        return True

    monkeypatch.setattr(email_service, "_send_email", fake_send)

    email_service.send_project_invitation_email(
        "invitee@example.com",
        "Invitee Name",
        "Inviter Name",
        "Example Project",
        action_url="https://app.example.com/projects/invitations/123",
    )

    assert captured["to_email"] == "invitee@example.com"
    assert "View invitation" in captured["html"]
    assert "Accept invitation" not in captured["html"]
    assert "Reject invitation" not in captured["html"]
    assert "View invitation:" in captured["text"]
    assert "Accept invitation:" not in captured["text"]
    assert "Reject invitation:" not in captured["text"]
    assert "https://app.example.com/projects/invitations/123" in captured["html"]


def test_project_invitation_email_handles_missing_action_link(monkeypatch):
    monkeypatch.setattr(email_service, "is_configured", lambda: True)

    captured = {}

    def fake_send(to_email, subject, text_body, html_body):
        captured.update({"text": text_body, "html": html_body})
        return True

    monkeypatch.setattr(email_service, "_send_email", fake_send)

    email_service.send_project_invitation_email(
        "invitee@example.com",
        "Invitee Name",
        "Inviter Name",
        "Example Project",
        action_url=None,
    )

    assert "View invitation" not in captured["html"]
    assert "View invitation:" not in captured["text"]

import pytest

from modules.auth.email_service import email_service


def test_project_invitation_email_includes_accept_reject(monkeypatch):
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
        accept_url="https://app.example.com/projects/invitations/123?response=accept",
        reject_url="https://app.example.com/projects/invitations/123?response=reject",
    )

    assert captured["to_email"] == "invitee@example.com"
    assert "Accept invitation" in captured["html"]
    assert "Reject invitation" in captured["html"]
    assert "Accept invitation:" in captured["text"]
    assert "Reject invitation:" in captured["text"]
    assert "https://app.example.com/projects/invitations/123?response=accept" in captured["html"]
    assert "https://app.example.com/projects/invitations/123?response=reject" in captured["html"]


@pytest.mark.parametrize(
    "accept_url,reject_url,expected_snippet",
    [
        (None, None, "View invitation"),
        ("https://app.example.com/accept", None, "Accept invitation"),
        (None, "https://app.example.com/reject", "Reject invitation"),
    ],
)
def test_project_invitation_email_handles_optional_links(monkeypatch, accept_url, reject_url, expected_snippet):
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
        action_url="https://app.example.com/projects/invitations/123",
        accept_url=accept_url,
        reject_url=reject_url,
    )

    assert expected_snippet in captured["html"]
    if accept_url:
        assert "Accept invitation:" in captured["text"]
    else:
        assert "Accept invitation:" not in captured["text"]

    if reject_url:
        assert "Reject invitation:" in captured["text"]
    else:
        assert "Reject invitation:" not in captured["text"]

"""Only admins can upload, manage documents, and see jobs.

A non-admin is refused with 403 and sent from the admin pages back to chat. A
signed-out visitor is refused with 401 and sent to Auth0.
"""

import pytest
from fastapi.testclient import TestClient

from api.auth.users import viewer
from api.server import app
from test.factories import TEST_USER_ID

ADMIN_ENDPOINTS = [
    ("post", "/api/upload"),
    ("get", "/api/documents"),
    ("get", "/api/jobs"),
    ("get", "/api/jobs/current"),
    ("get", "/api/models"),
]


@pytest.fixture
def signed_in_as():
    def sign_in(who):
        app.dependency_overrides[viewer] = lambda: who
        return TestClient(app, follow_redirects=False)

    yield sign_in
    app.dependency_overrides.clear()


@pytest.mark.parametrize("method, path", ADMIN_ENDPOINTS)
def test_a_non_admin_is_refused(signed_in_as, method, path):
    client = signed_in_as((TEST_USER_ID, False))

    assert client.request(method, path).status_code == 403


@pytest.mark.parametrize("method, path", ADMIN_ENDPOINTS)
def test_a_signed_out_visitor_is_asked_to_log_in(signed_in_as, method, path):
    client = signed_in_as(None)

    assert client.request(method, path).status_code == 401


@pytest.mark.parametrize("page", ["/upload", "/documents", "/jobs"])
def test_a_non_admin_is_sent_back_to_chat(signed_in_as, page):
    client = signed_in_as((TEST_USER_ID, False))

    assert client.get(page).headers["location"] == "/"


@pytest.mark.parametrize("page", ["/upload", "/documents", "/jobs"])
def test_a_signed_out_visitor_is_sent_to_auth0_and_back(signed_in_as, page):
    client = signed_in_as(None)

    location = client.get(page).headers["location"]

    assert location == f"/auth/login?returnTo={page}"

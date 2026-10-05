"""Tests for the delete-book task, written before the route exists.

The implementer must make these pass. They are copied into the target
repository's tests folder by the pipeline and committed with the change.
"""

import pytest

from app import create_app


@pytest.fixture
def client():
    """A test client on a fresh app, so each test starts with an empty reading list."""
    app = create_app()
    # TESTING makes Flask raise errors instead of returning a 500 page, which keeps failures readable.
    app.config["TESTING"] = True
    return app.test_client()


def test_delete_book(client):
    """Deleting an existing book returns 204 and removes it from the list."""
    # Create a book first so there is something to delete, and take its id from the response.
    book_id = client.post("/api/books", json={"title": "Dune"}).get_json()["id"]
    response = client.delete(f"/api/books/{book_id}")
    assert response.status_code == 204
    # The list must be empty again, which proves the book was removed rather than just acknowledged.
    assert client.get("/api/books").get_json() == []


def test_delete_missing_book(client):
    """Deleting an id that does not exist returns 404 with the agreed error body."""
    response = client.delete("/api/books/999")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Book not found"}

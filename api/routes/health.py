"""Liveness for uptime checks: the process answers and Postgres does too."""

from fastapi import APIRouter

from db.pool import connection

router = APIRouter()


@router.get("/health")
def health():
    with connection() as conn:
        conn.execute("select 1")
    return {"ok": True}

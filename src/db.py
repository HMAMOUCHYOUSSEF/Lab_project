"""Shared database connection helper."""
import os
import psycopg2


def connect():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "labdb"),
        user=os.getenv("PGUSER", "lab"),
        password=os.getenv("PGPASSWORD", "labpass"),
    )
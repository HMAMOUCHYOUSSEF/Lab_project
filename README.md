# Lab Project

Phase 0: Setup and Assumptions.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Start PostgreSQL: `docker compose up -d db`
4. Test:

```bash
docker exec lab_pg psql -U lab -d labdb -c "SELECT version();"
python -c "import pandas, numpy, pyarrow, faker, duckdb, psycopg2; print('imports OK')"


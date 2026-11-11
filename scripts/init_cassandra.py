"""Create the configured Cassandra keyspace and operational tables."""
from src.storage.cassandra_store import ensure_schema

if __name__ == "__main__":
    ensure_schema()
    print("Cassandra keyspace and tables are ready.")

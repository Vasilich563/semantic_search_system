from sqlalchemy import create_engine, text

EMBEDDING_DIM = 768


def actions_on_snippet_level(level):
    connection.execute(
        text(f"""
            CREATE TABLE IF NOT EXISTS triple_s.snippet_level{level}(
                snippet_name TEXT PRIMARY KEY,
                document_path TEXT NOT NULL,
                document_name TEXT NOT NULL,
                snippet TEXT NOT NULL,
                embedding vector(:dim)
            );
        """), {"dim": EMBEDDING_DIM}
    )

    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS
                snippet_level{level}_document_name_hash_index 
                ON triple_s.snippet_level{level} USING HASH (document_name);
        """)
    )

    connection.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS
                snippet_level{level}_embedding_hnsw_index 
                ON triple_s.snippet_level{level} USING hnsw (embedding vector_cosine_ops);
        """)
    )



if __name__ == "__main__":
    db_engine = create_engine("postgresql://postgres:ValhalaWithZolinks@localhost:5432/postgres")

    with db_engine.begin() as connection:
        connection.execute(text("""CREATE EXTENSION vector;"""))

        connection.execute(text("""CREATE SCHEMA IF NOT EXISTS triple_s;"""))

        actions_on_snippet_level(1)

        actions_on_snippet_level(2)

        actions_on_snippet_level(3)





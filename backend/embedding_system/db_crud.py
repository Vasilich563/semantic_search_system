from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from make_db import LEVEL_TABLE_NAME_PREFIX, SCHEMA_NAME


class DBCrud:

    __insert_template = """
        INSERT INTO {schema_name}.{table_name} 
            (snippet_name, document_path, document_name, snippet, embedding) 
            VALUES (:snippet_name, :document_path, :document_name, :snippet, :embedding)}
    """


    def __init__(self, db_engine: AsyncEngine):
        self._db_engine: AsyncEngine = db_engine
        self._level1_name = f"{LEVEL_TABLE_NAME_PREFIX}1"
        self._level2_name = f"{LEVEL_TABLE_NAME_PREFIX}2"
        self._level3_name = f"{LEVEL_TABLE_NAME_PREFIX}3"


    async def write_snippet_rows(self, list_of_rows):
        """
        :param list_of_rows: List[
            Dict[
                "snippet_name": str,
                "document_path": str,
                "document_name": str,
                "snippet": str,
                "embedding": List[float]
            ]
        ]
        :return: None
        """
        list_of_rows = self._unpack_snippets(document_path, document_name, snippet_bounds_batch, embeddings_batch)

        async with self._db_engine.begin() as connection:
            await connection.execute(
                text(self.__insert_template.format(SCHEMA_NAME, self._level1_name)),
                list_of_rows
            )


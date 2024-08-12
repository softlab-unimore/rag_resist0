import os
import logging
import psycopg2

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(filename="./log/bper.log", level=logging.INFO)
logger = logging.getLogger("bper.connector")

class PgVectorConnector:
    def __init__(self):
        username = os.environ["POSTGRES_USER"]
        password = os.environ["POSTGRES_PASSWORD"]
        db = os.environ["POSTGRES_DB"]
        port = os.environ["POSTGRES_PORT"]
        self.connection = f"postgresql+psycopg://{username}:{password}@localhost:{port}/{db}"
    
    def get_connection(self):
        return self.connection
    
    def start_db_connection(self):
        connection = self.connection.replace("+psycopg", "")
        return psycopg2.connect(connection)

    def close_db_connection(self, conn):
        conn.close()

    def get_existing_ids(self, conn, ids):
        formatted_ids = ', '.join(f"'{id}'" for id in ids)
        query = f"SELECT id FROM {os.environ['POSTGRES_EMB_TABLE_NAME']} WHERE id IN ({formatted_ids});"

        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
        return [row[0] for row in result]
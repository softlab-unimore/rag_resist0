import os
import logging

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
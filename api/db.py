from config import settings
import json
from google.cloud.sql.connector import Connector
import sqlalchemy
import structlog

logger = structlog.get_logger()


class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            # Initialize the Cloud SQL Python Connector
            connector = Connector()

            # Function to return a database connection
            def getconn():
                conn = connector.connect(
                    settings.POSTGRES_INSTANCE_CONNECTION,
                    "pg8000",
                    user="postgres",
                    password=settings.POSTGRES_DB_PASSWORD,
                    db="postgres",
                )
                return conn

            try:
                # Create a SQLAlchemy engine object
                cls._instance = sqlalchemy.create_engine(
                    "postgresql+pg8000://", creator=getconn
                )
            except:
                pass
        return cls._instance


def get_db():
    return Database()


class MatchResult:
    question_id: str
    similarity: float

    def __init__(self, question_id: str, similarity: float):
        self.question_id = question_id
        self.similarity = float(similarity)


def match_documents(embedding):
    try:
        with get_db().connect() as connection:
            result = connection.execute(
                sqlalchemy.text(
                    "SELECT match_documents(:embedding, :match_threshold, :match_count)"
                ),
                {
                    "embedding": json.dumps(embedding),
                    "match_threshold": float(settings.VECTOR_MATCH_THRESHOLD),
                    "match_count": int(settings.VECTOR_MATCH_COUNT),
                },
            )
            return [MatchResult(obj[0][1], obj[0][2]) for obj in result.all()]
    except Exception as e:
        logger.error(f"Error matching embedding on Cloud SQL: {e}")
        return []


class QA:
    id: str
    question: str
    answer: str

    def __init__(self, id: str, question: str, answer: str):
        self.id = id
        self.question = question
        self.answer = answer


from sqlalchemy import Table, Column, String, MetaData, select

metadata = MetaData()

qas = Table(
    "qas",
    metadata,
    Column("id", String, primary_key=True),
    Column("question", String),
    Column("answer", String),
)


def get_qas(ids):
    try:
        query = select(qas).where(qas.c.id.in_(ids))

        # Execute the query
        with get_db().connect() as connection:
            result = connection.execute(query)
            data = result.mappings().all()

        return [QA(**obj) for obj in data]

    except Exception as e:
        logger.error(f"Error loading QAs from Cloud SQL: {e}")
        return []

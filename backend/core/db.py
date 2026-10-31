from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.models.database import Base

class DatabaseManager:
    """
    Handles the immutable storage of experiments.
    """
    def __init__(self, db_url: str = "sqlite:///./signalforge.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def save_experiment(self, experiment_data: Dict[str, Any]):
        """
        Saves an experiment record. Ensures configuration is immutable.
        """
        from backend.models.database import Experiment
        session = self.get_session()
        try:
            exp = Experiment(**experiment_data)
            session.merge(exp) # Update if exists, else create
            session.commit()
        finally:
            session.close()

    def get_experiment(self, experiment_id: str):
        from backend.models.database import Experiment
        session = self.get_session()
        try:
            return session.query(Experiment).filter(Experiment.id == experiment_id).first()
        finally:
            session.close()

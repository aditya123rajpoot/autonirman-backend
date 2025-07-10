from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Get DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# ✅ Debug print (optional)
print("DATABASE_URL from .env =", DATABASE_URL)

# ✅ Check if it's None (optional but useful)
if not DATABASE_URL:
    raise Exception("DATABASE_URL is missing. Check your .env file.")

# ✅ Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

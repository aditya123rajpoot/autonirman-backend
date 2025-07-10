from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth
from app.database import engine, Base

app = FastAPI()

# Allow frontend to access backend (Vercel <-> Railway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables
Base.metadata.create_all(bind=engine)

# Include routes
app.include_router(auth.router)

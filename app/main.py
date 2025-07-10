# app/main.py

import os
import dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, layout_smart, vastu
from app.database import engine, Base

# ✅ Load environment variables from .env
dotenv.load_dotenv()

# ✅ Initialize FastAPI app
app = FastAPI(title="Auto Nirman Backend")

# ✅ Enable CORS for frontend (adjust domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with actual domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Create DB tables from SQLAlchemy models
Base.metadata.create_all(bind=engine)

# ✅ Register API routes
app.include_router(auth.router, prefix="/api")
app.include_router(vastu.router, prefix="/api")
app.include_router(layout_smart.router, prefix="/api")

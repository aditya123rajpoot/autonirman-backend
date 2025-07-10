from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from app.vastu_engine import vastu_compliance_score

router = APIRouter()

class LayoutRequest(BaseModel):
    layout: Dict[str, str]

@router.post("/vastu-score")
def get_vastu_score(data: LayoutRequest):
    return vastu_compliance_score(data.layout)

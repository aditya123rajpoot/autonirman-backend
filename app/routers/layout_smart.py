import os
import json
import redis.asyncio as redis
import httpx

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, List, Dict, Any, Union
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  # ✅ Updated import
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from loguru import logger
from prometheus_client import Histogram, Counter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
# from groq.error import APIError, BadRequestError

# ---------------------
# 📊 Prometheus Metrics
# ---------------------
REQUEST_LATENCY = Histogram(
    "layout_smart_latency_seconds", "Latency for smart layout endpoint"
)

FALLBACK_USED = Counter(
    "layout_fallback_count", "Number of times fallback (GPT-4) was used due to Groq failure"
)

router = APIRouter(
    prefix="/api/v1/layout",
    tags=["Smart Layout"]
)

# ---------------------
# 📅 Input Schema
# ---------------------
class SmartLayoutRequest(BaseModel):
    total_builtup_area: int = Field(..., gt=100, example=2400)
    number_of_floors: int = Field(..., gt=0, example=2)
    shape_of_plot: Literal["Rectangular", "Square", "L-shaped", "Irregular"]
    your_city: str = Field(..., min_length=2, example="Lucknow")
    weather_in_your_city: Literal["Hot and Dry", "Cold", "Humid", "Moderate"]
    do_you_follow_vastu: Literal["Yes", "No"]
    session_id: Optional[str] = Field(None, description="Client session for memory")

    @validator("your_city")
    def title_case_city(cls, v):
        return v.strip().title()

# ---------------------
# 📄 Output Schema
# ---------------------
class Section(BaseModel):
    title: str
    content: Union[str, Dict[str, Any], List[str]]

class SmartLayoutResponse(BaseModel):
    session_id: Optional[str]
    plan_sections: List[Section]
    hints: List[str]
    metadata: Dict[str, Any]

# ---------------------
# 🔌 Plugin System
# ---------------------
class AdvisorPlugin:
    def apply(self, data: SmartLayoutRequest) -> Section:
        raise NotImplementedError

class VastuAdvisor(AdvisorPlugin):
    def apply(self, data):
        tips = []
        if data.do_you_follow_vastu == "Yes":
            tips = [
                "✅ Main entrance: East/North-East",
                "✅ Kitchen: South-East",
                "❌ Avoid slabs above pooja room"
            ]
        return Section(title="Vastu Compliance", content=tips)

class EcoAdvisor(AdvisorPlugin):
    def apply(self, data):
        pct = 0.1 if "Hot" in data.weather_in_your_city else 0.08
        return Section(
            title="Eco & Rainwater",
            content=f"Green cover: {int(data.total_builtup_area * pct)} sq ft; rain harvesting at rear setback"
        )

PLUGINS: List[AdvisorPlugin] = [VastuAdvisor(), EcoAdvisor()]

# ---------------------
# 🧠 LLM Factory
# ---------------------
def get_llm():
    key = os.getenv("GROQ_API_KEY")
    if key:
        logger.info("✅ Using Groq LLM: llama3-70b-8192")
        return ChatGroq(api_key=key, model_name="llama3-70b-8192", temperature=0.6)
    logger.warning("⚠️ Falling back to OpenAI GPT-4")
    return ChatOpenAI(model="gpt-4", temperature=0.6)

# ---------------------
# 🧠 LLM Prompt & Chain
# ---------------------
_PROMPT = PromptTemplate(
    input_variables=["area", "floors", "shape", "city", "weather", "vastu"],
    template="""
You are a world‎-class architect AI. Based on:
• Area: {area} sq ft
• Floors: {floors}
• Plot: {shape}
• City: {city} (Climate: {weather})
• Vastu: {vastu}

Return JSON with sections: Layout Plan, Setbacks, Parking, Climate Adaptation. 
Be concise and user‎-friendly.
"""
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(Exception)
)
async def call_llm(chain: LLMChain, vars: dict) -> str:
    try:
        return await chain.arun(vars)
    except Exception as e:
        logger.warning("⚠️ Groq failed. Switching to OpenAI GPT-4. Reason: {}", str(e))
        FALLBACK_USED.inc()
        fallback_llm = ChatOpenAI(model="gpt-4", temperature=0.6)
        fallback_chain = LLMChain(llm=fallback_llm, prompt=_PROMPT)
        return await fallback_chain.arun(vars)

# ---------------------
# 🚀 Endpoint
# ---------------------
@router.post(
    "/smart",
    response_model=SmartLayoutResponse,
    summary="Generate an AI-driven smart layout"
)
async def smart_layout(
    request: Request,
    data: SmartLayoutRequest,
    llm=Depends(get_llm)
):
    start = REQUEST_LATENCY.time()
    logger.bind(endpoint="layout_smart", session=data.session_id).info("Request start")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)

    memory_key = f"layout_mem:{data.session_id}"
    history = await redis_client.lrange(memory_key, 0, -1) if data.session_id else []

    chain = LLMChain(llm=llm, prompt=_PROMPT)
    cache_key = f"layout_cache:{json.dumps(data.dict(), sort_keys=True)}"
    cached = await redis_client.get(cache_key)

    if cached:
        response_text = cached
        logger.info("✅ Cache hit")
    else:
        variables = {
            "area": data.total_builtup_area,
            "floors": data.number_of_floors,
            "shape": data.shape_of_plot,
            "city": data.your_city,
            "weather": data.weather_in_your_city,
            "vastu": data.do_you_follow_vastu
        }

        logger.info("📤 Sending to LLM: {}", variables)

        try:
            response_text = await call_llm(chain, variables)
            logger.info("✅ LLM raw response: {}", response_text)
        except Exception as e:
            logger.error("❌ LLM call failed: {}", str(e))
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="LLM service unavailable")

        await redis_client.set(cache_key, response_text, ex=3600)
        logger.info("🧠 Response cached")

    try:
        parsed = json.loads(response_text)
        sections = [Section(**sec) for sec in parsed.get("sections", [])]
    except Exception:
        sections = [Section(title="Smart Layout", content=response_text)]

    for plugin in PLUGINS:
        sections.append(plugin.apply(data))

    duration = start()
    logger.bind(duration=duration).info("✅ Request complete")

    return SmartLayoutResponse(
        session_id=data.session_id,
        plan_sections=sections,
        hints=["You can refine by adjusting area or floors."],
        metadata={"cacheUsed": bool(cached), "duration_s": duration}
    )

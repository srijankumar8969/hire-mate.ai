from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import run_career_agent, resume_career_agent

# Kept from the original project to allow the synchronous agent functions to
# call async MCP helpers inside FastAPI.
import nest_asyncio

nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="HireMate AI",
    description=(
        "LangGraph Multi-Agent Career/Job Application Assistant with "
        "job-search, resume-matcher, cover-letter, and formatter agents, "
        "a Supervisor, input Guardrails, Human-in-the-Loop review, and a "
        "FastAPI frontend."
    ),
    version="2.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class ApplicationRequest(BaseModel):
    message: str
    resume_text: str = ""
    job_description: str = ""
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/apply")
async def apply(request_data: ApplicationRequest):
    try:
        user_message = request_data.message.strip()
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        result = run_career_agent(
            user_input=user_message,
            resume_text=request_data.resume_text,
            job_description=request_data.job_description,
            thread_id=request_data.thread_id,
        )
        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )
    except Exception as exc:
        print("ERROR:", exc)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.post("/api/apply/approve")
async def approve_application(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        result = resume_career_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )
        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )
    except Exception as exc:
        print("APPROVAL ERROR:", exc)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "HireMate AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "job_search_agent",
            "resume_matcher_agent",
            "cover_letter_agent",
            "formatter_agent",
            "human_in_the_loop",
        ],
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

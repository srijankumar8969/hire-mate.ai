import os

import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from job_mcp_client import (
    job_search_mcp_call,
    extract_job_criteria,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Postgres database URL to .env"
        )
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"
    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM
# =========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)

# =========================
# State
# =========================
class CareerState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    resume_text: str
    job_description: str  # optional: paste a specific JD instead of searching

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    job_criteria: dict[str, Any]
    supervisor_reasoning: str

    # Specialist results
    job_search_results: str
    match_analysis: str
    match_score: str
    cover_letter: str
    formatted_output: str

    # HITL state
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str
    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "job_search_agent",
    "resume_matcher_agent",
    "cover_letter_agent",
    "formatter_agent",
}

AGENT_ORDER = [
    "job_search_agent",
    "resume_matcher_agent",
    "cover_letter_agent",
    "formatter_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")
    return json.loads(text[start : end + 1])


def _empty_criteria() -> dict[str, Any]:
    return {
        "target_role": "",
        "location": "",
        "seniority": "",
        "keywords": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: CareerState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to career or job-application
assistance. Valid requests can include job searching, resume feedback,
resume-to-job matching, cover letter writing, application formatting, or
general career/interview advice. Block clearly unrelated requests and
requests asking for harmful, illegal, or deceptive content (e.g. fabricating
credentials, plagiarizing someone else's resume). Do not block a valid
request merely because some details are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the workflow.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a career/job-application "
            "assistant application. Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "HireMate AI can only help with job-search and application "
            "requests. Please ask about job searching, resume matching, "
            "cover letters, or application formatting."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "job_criteria": _empty_criteria(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent career/job-application system.

Choose only the specialist agents needed for the request.

Available agents:
- job_search_agent: finds live job postings matching a role/location
- resume_matcher_agent: compares a resume against a job description and scores fit
- cover_letter_agent: writes a tailored cover letter
- formatter_agent: assembles the final polished application package and must
  always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["job_search_agent", "resume_matcher_agent", "cover_letter_agent", "formatter_agent"],
  "job_criteria": {{
    "target_role": "",
    "location": "",
    "seniority": "",
    "keywords": []
  }},
  "reasoning": ""
}}

User request:
{query}

Resume provided: {"yes" if state.get("resume_text") else "no"}
Job description provided: {"yes" if state.get("job_description") else "no"}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to career-assistant specialist agents. "
            "Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # Don't try to match/write a cover letter with no resume on hand.
        if not state.get("resume_text"):
            selected_agents = [
                a for a in selected_agents
                if a not in ("resume_matcher_agent", "cover_letter_agent")
            ]

        # The formatter agent integrates whichever specialist results were selected.
        if "formatter_agent" not in selected_agents:
            selected_agents.append("formatter_agent")

        criteria = _empty_criteria()
        parsed_criteria = parsed.get("job_criteria", {})
        if isinstance(parsed_criteria, dict):
            criteria.update(parsed_criteria)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Safe fallback: run the full pipeline (minus matcher/cover-letter if
        # there's no resume to work with).
        selected_agents = AGENT_ORDER.copy()
        if not state.get("resume_text"):
            selected_agents = [
                a for a in selected_agents
                if a not in ("resume_matcher_agent", "cover_letter_agent")
            ]
        criteria = extract_job_criteria(query)
        reasoning = (
            "Supervisor parsing failed, so the full job-application workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "job_criteria": criteria,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: CareerState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the career-assistant input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Job Search Agent (job board API)
# =========================
def job_search_agent(state: CareerState):
    print("\nINSIDE JOB SEARCH AGENT\n")
    criteria = state.get("job_criteria", {}) or {}
    role = criteria.get("target_role") or state["user_query"]
    location = criteria.get("location", "")

    try:
        search_results = asyncio.run(job_search_mcp_call(role, location, limit=8))
        print("\nJOB RESULTS:", search_results)

        summary_prompt = f"""
User Request:
{state['user_query']}

Search Criteria:
{criteria}

Raw Job Listings (JSON):
{str(search_results)[:4000]}

Summarize the most relevant postings as a short numbered list. For each
posting include: title, company, location, job type, salary (if listed),
and the application URL. Note if results are sparse or unrelated to the
request.
"""
        response = llm.invoke(
            [
                SystemMessage(content="You are an expert job-search assistant."),
                HumanMessage(content=summary_prompt),
            ]
        )
        job_search_results = str(response.content)
    except Exception as exc:
        print(f"JOB SEARCH AGENT MCP ERROR: {type(exc).__name__}: {exc}", flush=True)
        job_search_results = (
            "Live job search is temporarily unavailable. Provide general "
            "guidance on where to look for this role and clearly label it "
            "as non-live advice."
        )

    return {
        "job_search_results": job_search_results,
        "messages": [AIMessage(content="Job search results generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Resume Matcher Agent
# =========================
def resume_matcher_agent(state: CareerState):
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "") or state.get("job_search_results", "")

    if not resume_text:
        return {
            "match_analysis": "No resume was provided, so a fit analysis could not be run.",
            "match_score": "N/A",
            "messages": [AIMessage(content="Resume matching skipped: no resume provided.")],
        }

    prompt = f"""
Compare this resume against the target job and produce a fit analysis.

Resume:
{resume_text[:6000]}

Target Job Description / Search Results:
{job_description[:4000]}

Job Criteria:
{state.get('job_criteria', {})}

Return:
1. Overall match score out of 100 (label it clearly as "Match Score: NN/100")
2. Strongest matching skills/experience
3. Gaps or missing keywords compared to the job requirements
4. 3-5 concrete suggestions to improve the resume for this specific role
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert resume-to-job fit analyst and ATS specialist."),
            HumanMessage(content=prompt),
        ]
    )
    analysis_text = str(response.content)

    match_score = "N/A"
    for line in analysis_text.splitlines():
        if "match score" in line.lower():
            match_score = line.strip()
            break

    return {
        "match_analysis": analysis_text,
        "match_score": match_score,
        "messages": [AIMessage(content="Resume match analysis generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Cover Letter Agent
# =========================
def cover_letter_agent(state: CareerState):
    resume_text = state.get("resume_text", "")

    if not resume_text:
        return {
            "cover_letter": "No resume was provided, so a cover letter could not be drafted.",
            "messages": [AIMessage(content="Cover letter skipped: no resume provided.")],
        }

    prompt = f"""
Write a tailored, professional cover letter (under 400 words).

User Request:
{state['user_query']}

Resume:
{resume_text[:6000]}

Target Job Description / Search Results:
{state.get('job_description', '') or state.get('job_search_results', '')[:3000]}

Resume-Fit Analysis:
{state.get('match_analysis', '')}

Guidelines:
- Address it generically (e.g. "Dear Hiring Manager") unless a company name is clear.
- Lead with the candidate's strongest, most relevant experience.
- Reference specific skills/keywords from the job description or job criteria.
- Keep the tone confident and specific, not generic filler.
- Do not fabricate experience that isn't in the resume.
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert cover letter writer."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "cover_letter": str(response.content),
        "messages": [AIMessage(content="Cover letter drafted.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Formatter Agent - assembles the draft application package
# =========================
def formatter_agent(state: CareerState):
    prompt = f"""
Assemble a clean, well-formatted draft application package in Markdown.

User Request:
{state['user_query']}

Job Criteria:
{state.get('job_criteria', {})}

Job Search Results:
{state.get('job_search_results', '')}

Resume-Fit Analysis:
{state.get('match_analysis', '')}

Cover Letter:
{state.get('cover_letter', '')}

Format the draft using these sections (omit a section only if there is
genuinely nothing to put in it):
1. Job Matches
2. Resume Fit Analysis
3. Tailored Cover Letter
4. Suggested Next Steps

Make it easy to scan and ready for human review before finalizing.
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert career-application formatter."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated application draft. Approve it to "
        "create the final polished package, or provide feedback for revision."
    )

    return {
        "formatted_output": str(response.content),
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft application package created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: CareerState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this application draft?",
            "draft_output": state.get("formatted_output", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent
# =========================
def final_agent(state: CareerState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final job-application response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Job Criteria:
{state.get('job_criteria', {})}

Job Search Results:
{state.get('job_search_results', '')}

Resume Fit Analysis:
{state.get('match_analysis', '')}

Cover Letter:
{state.get('cover_letter', '')}

Draft Package:
{state.get('formatted_output', '')}

Format the final answer beautifully using these sections:
1. Application Summary
2. Job Matches
3. Resume Fit & Match Score
4. Tailored Cover Letter
5. Next Steps / Application Checklist

Important:
- Be clear and practical.
- Note when live job-board results were unavailable or sparse.
- Do not fabricate credentials or experience not present in the resume.
- Incorporate the human feedback when a revision was requested.
"""
    response = llm.invoke(
        [
            SystemMessage(content="You are a professional AI career-application assistant."),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": str(response.content),
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "job_search_agent": "job_search_agent",
    "resume_matcher_agent": "resume_matcher_agent",
    "cover_letter_agent": "cover_letter_agent",
    "formatter_agent": "formatter_agent",
}


def _selected_agents(state: CareerState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: CareerState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"
    selected = _selected_agents(state)
    return selected[0] if selected else "formatter_agent"


def route_after_agent(current_agent: str):
    def route(state: CareerState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent
        return "formatter_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(CareerState)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("job_search_agent", job_search_agent)
graph.add_node("resume_matcher_agent", resume_matcher_agent)
graph.add_node("cover_letter_agent", cover_letter_agent)
graph.add_node("formatter_agent", formatter_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
graph.add_conditional_edges(
    "job_search_agent", route_after_agent("job_search_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "resume_matcher_agent", route_after_agent("resume_matcher_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "cover_letter_agent", route_after_agent("cover_letter_agent"), ROUTE_MAP
)
graph.add_edge("formatter_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

career_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message

    interrupt_payload = _interrupt_payload(result)
    if interrupt_payload:
        answer = interrupt_payload.get("draft_output") or result.get(
            "formatted_output", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "job_search_results": result.get("job_search_results", ""),
        "match_analysis": result.get("match_analysis", ""),
        "match_score": result.get("match_score", ""),
        "cover_letter": result.get("cover_letter", ""),
        "formatted_output": (
            interrupt_payload.get("draft_output", "")
            if interrupt_payload
            else result.get("formatted_output", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "job_criteria": result.get("job_criteria", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_career_agent(
    user_input: str,
    resume_text: str = "",
    job_description: str = "",
    thread_id: str | None = None,
):
    """Start a new job-application run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}
    result = career_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "resume_text": resume_text,
            "job_description": job_description,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "job_criteria": _empty_criteria(),
            "supervisor_reasoning": "",
            "job_search_results": "",
            "match_analysis": "",
            "match_score": "",
            "cover_letter": "",
            "formatted_output": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )
    return _serialize_result(result, thread_id)


def resume_career_agent(thread_id: str, approved: bool, feedback: str = ""):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a job-application run.")

    config = {"configurable": {"thread_id": thread_id}}
    result = career_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )
    return _serialize_result(result, thread_id)

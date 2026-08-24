import json
import os
import sys
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

# =========================================================
# Environment setup
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JOB_SERVER_PATH = BASE_DIR / "custom_job_mcp_server.py"


def _require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is missing. Add {name}=your_key to the project .env file."
        )
    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """Preserve the current environment and add any extra MCP env vars."""
    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env


# =========================================================
# LLM
# =========================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
)

# =========================================================
# MCP client - a single keyless job-board server, same shape
# as the original project's MultiServerMCPClient so it's easy
# to add real providers (Adzuna, JSearch, LinkedIn scraper, ...)
# as additional servers later.
# =========================================================
client = MultiServerMCPClient(
    {
        "jobsearch": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(JOB_SERVER_PATH)],
            "env": _subprocess_env(),
        },
    }
)


async def _get_server_tool(server_name: str, tool_name: str):
    if server_name == "jobsearch" and not JOB_SERVER_PATH.is_file():
        raise FileNotFoundError(f"Job search MCP server not found: {JOB_SERVER_PATH}")

    tools = await client.get_tools(server_name=server_name)
    tool = next((item for item in tools if item.name == tool_name), None)
    if tool is None:
        available_tools = ", ".join(sorted(item.name for item in tools)) or "none"
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )
    return tool


async def get_all_tools() -> None:
    """Test the job-search MCP server connection independently."""
    try:
        tools = await client.get_tools(server_name="jobsearch")
        tool_names = ", ".join(tool.name for tool in tools) or "no tools"
        print(f"jobsearch: OK -> {tool_names}")
    except Exception as exc:
        print(f"jobsearch: FAILED -> {type(exc).__name__}: {exc}")


# =========================================================
# Job-search MCP
# =========================================================
async def job_search_mcp_call(query: str, location: str = "", limit: int = 8):
    tool = await _get_server_tool("jobsearch", "search_jobs")
    return await tool.ainvoke({"query": query, "location": location, "limit": limit})


async def job_categories_mcp_call():
    tool = await _get_server_tool("jobsearch", "get_job_categories")
    return await tool.ainvoke({})


# =========================================================
# Job-criteria extractor (mirrors extract_destination in the
# original travel project)
# =========================================================
def extract_job_criteria(query: str) -> dict[str, Any]:
    prompt = f"""
Extract structured job-search criteria from this request.

Request:
{query}

Return strict JSON only using this schema:
{{
  "target_role": "",
  "location": "",
  "seniority": "",
  "keywords": []
}}
"""
    response = llm.invoke(prompt)
    text = str(response.content)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"target_role": query, "location": "", "seniority": "", "keywords": []}
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return {"target_role": query, "location": "", "seniority": "", "keywords": []}

    return {
        "target_role": str(parsed.get("target_role", "")).strip() or query,
        "location": str(parsed.get("location", "")).strip(),
        "seniority": str(parsed.get("seniority", "")).strip(),
        "keywords": parsed.get("keywords", []) if isinstance(parsed.get("keywords"), list) else [],
    }

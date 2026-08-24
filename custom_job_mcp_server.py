import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

mcp = FastMCP("Job Search MCP Server")

# Remotive is a free, keyless remote-job-board API. Swap this out for
# Adzuna/JSearch/LinkedIn/Indeed scraping later without touching backend.py -
# the agent only ever talks to the `search_jobs` / `get_job_categories` tools.
REMOTIVE_JOBS_URL = "https://remotive.com/api/remote-jobs"
REMOTIVE_CATEGORIES_URL = "https://remotive.com/api/remote-jobs/categories"
REQUEST_TIMEOUT_SECONDS = 20


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        details = ""
        failed_response = getattr(exc, "response", None)
        if failed_response is not None:
            details = f" Response: {failed_response.text[:500]}"
        raise RuntimeError(f"Job board request failed: {exc}.{details}") from exc


@mcp.tool()
def search_jobs(query: str, location: str = "", limit: int = 8) -> dict[str, Any]:
    """Search live remote job postings by keyword/role, optionally filtered by location."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    limit = max(1, min(limit, 20))
    data = _request_json(REMOTIVE_JOBS_URL, {"search": query, "limit": limit * 3})

    jobs = []
    for job in data.get("jobs", []):
        if location:
            candidate_location = str(job.get("candidate_required_location", "")).lower()
            if location.lower() not in candidate_location and "anywhere" not in candidate_location:
                continue
        jobs.append(
            {
                "id": job.get("id"),
                "title": job.get("title"),
                "company": job.get("company_name"),
                "category": job.get("category"),
                "location": job.get("candidate_required_location"),
                "job_type": job.get("job_type"),
                "salary": job.get("salary") or "Not listed",
                "url": job.get("url"),
                "description_snippet": (job.get("description") or "")[:800],
                "publication_date": job.get("publication_date"),
            }
        )
        if len(jobs) >= limit:
            break

    return {"query": query, "location": location, "count": len(jobs), "jobs": jobs}


@mcp.tool()
def get_job_categories() -> dict[str, Any]:
    """Return the list of job categories supported by the job board."""
    return _request_json(REMOTIVE_CATEGORIES_URL, {})


if __name__ == "__main__":
    # job_mcp_client.py launches this as a stdio subprocess.
    mcp.run(transport="stdio")

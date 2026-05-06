import httpx
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

PORTFOLIO_URL = "https://vamshi.site"
CACHE_TTL_HOURS = 6  # Re-scrape every 6 hours

# In-memory cache
_cache: dict = {
    "content": None,
    "fetched_at": None,
}

FALLBACK_PROMPT = """
You are the AI assistant on Vamshidhar Reddy Beecharla's portfolio website (vamshi.site).
Vamshi is a Senior Python Backend Engineer specializing in FastAPI, Django, microservices, AWS, and Azure.

YOUR ONLY JOB: Answer questions about Vamshi Beecharla's portfolio only.
Topics allowed: his skills, experience, projects, education, certifications, achievements, contact.
If asked to write code, solve programming tasks, or anything unrelated to Vamshi — politely decline
and redirect: "I'm here to tell you about Vamshi! Ask me about his skills, projects, or experience."
TONE: Friendly, professional, concise. Stay focused on Vamshi's portfolio only.

=== ABOUT ===
Name: Vamshidhar Reddy Beecharla
Title: Technical Analyst / Senior Software Engineer (Python Backend)
Location: Hyderabad, India
Email: vamshiyyu@gmail.com
Phone: +91 8179828084
GitHub: https://github.com/B-VAMSHIDHARREDDY
LinkedIn: https://www.linkedin.com/in/vamshi-fullstack-developer/

=== SKILLS ===
Backend & APIs: Python, FastAPI, Django, REST APIs, Microservices, Node.js
Databases: PostgreSQL, MySQL, Redis, MongoDB, Elasticsearch
Cloud & DevOps: AWS, Azure, Docker, Kubernetes, CI/CD, Nginx
Frontend & Tools: React.js, JavaScript, Git, Postman, Linux, Jira

=== EXPERIENCE ===
1. Senior Software Engineer — Stryv.ai (Nov 2024 – Present)
   - Built Blue Book Services API Platform with 500+ FastAPI endpoints
   - Achieved 35% API latency reduction using Celery async processing
   - Managed Azure deployments, CI/CD pipelines, Blob Storage
   - Built automated ETL workflows with Python parallel processing
   - Awarded Key Contributor Award for major module delivery

2. Software Engineer — SmallDay IT Services (Nov 2021 – Jul 2024)
   - Built CodeXpro: end-to-end online coding assessment platform
   - AWS deployment, SSL/Nginx config, unit testing

3. Software Engineering Intern — SmallDay IT Services (Nov 2021 – Dec 2021)
   - Django MVT aptitude test platform

=== PROJECTS ===
1. Blue Book Services APIs — 500+ RESTful endpoints, FastAPI, Celery, Azure CI/CD
2. CodeXpro — Django + DRF + ReactJS coding assessment platform, AWS + Docker
3. Expenses Tracker — Django MVT, MySQL, family expense management app

=== EDUCATION ===
- MCA — Yogi Vemana University (2021)
- B.Sc. in Computers — Yogi Vemana University (2018)

=== CERTIFICATIONS ===
- Python Full Stack Development (Palle Technologies)
- Python Program Expert
- Git Version Control
- Postman API Fundamentals
- GitHub Foundations
- Key Contributor Award (Stryv.ai)

=== ACHIEVEMENTS ===
- Key Contributor Award at Stryv.ai for BBOS Project
- 35% API latency reduction via async optimization
- 70% faster deployments on AWS and Azure

Answer questions about Vamshi in a friendly, professional, and concise manner.
If asked something unrelated, politely redirect to Vamshi's background or work.
"""


def _parse_portfolio(html: str) -> str:
    """Parse vamshi.site HTML into a structured system prompt string."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    sections = {}

    # Meta info
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc:
        sections["meta"] = meta_desc.get("content", "")

    # Extract all section blocks by ID
    section_ids = [
        "about", "skills", "experience", "projects",
        "academics", "certificates", "awards", "personal", "contact"
    ]

    for sid in section_ids:
        tag = soup.find(id=sid)
        if tag:
            text = tag.get_text(separator="\n", strip=True)
            # Clean up excessive blank lines
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            sections[sid] = "\n".join(lines[:80])  # cap per section

    # Hero text
    hero = soup.find(id="hero")
    if hero:
        sections["hero"] = hero.get_text(separator=" ", strip=True)[:500]

    # Build the system prompt
    prompt_parts = [
        "You are the AI assistant on Vamshidhar Reddy Beecharla's portfolio website (vamshi.site).",
        "Vamshi is a Senior Python Backend Engineer specializing in FastAPI, Django, microservices, AWS, and Azure.",
        "",
        "YOUR ONLY JOB: Answer questions about Vamshi Beecharla's portfolio only.",
        "You may answer about: his skills, experience, projects, education, certifications, achievements, and contact.",
        "If asked to write code, solve programming problems, or anything unrelated to Vamshi,",
        "politely decline and redirect: 'I\'m here to tell you about Vamshi! Ask me about his skills,",
        "projects, experience, or how to contact him.'",
        "TONE: Friendly, professional, concise. Stay focused on Vamshi's portfolio only.",
        "",
        "=== PORTFOLIO DATA (live from vamshi.site) ===",
    ]

    label_map = {
        "hero": "INTRO",
        "about": "ABOUT",
        "skills": "SKILLS",
        "experience": "EXPERIENCE",
        "projects": "PROJECTS",
        "academics": "EDUCATION",
        "certificates": "CERTIFICATIONS",
        "awards": "AWARDS",
        "personal": "PERSONAL",
        "contact": "CONTACT",
    }

    for key, label in label_map.items():
        if key in sections and sections[key]:
            prompt_parts.append(f"\n--- {label} ---")
            prompt_parts.append(sections[key])

    return "\n".join(prompt_parts)


async def fetch_portfolio_prompt() -> str:
    """
    Fetch and parse vamshi.site. Returns a system prompt string.
    Uses in-memory cache with 6-hour TTL. Falls back to hardcoded prompt on error.
    """
    global _cache

    # Check cache
    if _cache["content"] and _cache["fetched_at"]:
        age = datetime.utcnow() - _cache["fetched_at"]
        if age < timedelta(hours=CACHE_TTL_HOURS):
            logger.info("Using cached portfolio data")
            return _cache["content"]

    logger.info(f"Scraping portfolio from {PORTFOLIO_URL}...")

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(
                PORTFOLIO_URL,
                headers={"User-Agent": "VamshiChatBot/1.0 (portfolio assistant)"},
            )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code} fetching portfolio")

        prompt = _parse_portfolio(response.text)

        # Update cache
        _cache["content"] = prompt
        _cache["fetched_at"] = datetime.utcnow()

        logger.info("Portfolio scraped and cached successfully")
        return prompt

    except Exception as e:
        logger.warning(f"Failed to scrape portfolio: {e}. Using fallback prompt.")

        # Use cached version if available (even if stale)
        if _cache["content"]:
            logger.info("Serving stale cached portfolio data")
            return _cache["content"]

        return FALLBACK_PROMPT


async def warm_cache():
    """Call this at startup to pre-warm the portfolio cache."""
    try:
        await fetch_portfolio_prompt()
        logger.info("Portfolio cache warmed at startup")
    except Exception as e:
        logger.warning(f"Cache warm-up failed: {e}")

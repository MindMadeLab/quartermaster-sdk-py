"""Example 25 — Customer enrichment: static web scrapes + dynamic agent research.

Demonstrates composing sub-graphs with the main graph to enrich sparse
customer data (just a name + country) into a full contact profile.

Architecture::

    ┌─── web_scrape_graph (sub-graph, reused 3x) ──────────┐
    │  static(url) → agent(scrape_url) → extract info        │
    └───────────────────────────────────────────────────────┘

    ┌─── main graph ────────────────────────────────────────┐
    │  user("Customer name + country")                       │
    │  → web_scrape_graph(bizi.si)                           │
    │  → web_scrape_graph(google maps)                       │
    │  → web_scrape_graph(company website)                   │
    │  → agent(tools=[search_web, scrape_url])               │
    │  → instruction("Compile Notes")                        │
    └───────────────────────────────────────────────────────┘
    → qm.instruction_form(schema=CustomerProfile)

Reads provider config from .env:
    OLLAMA_API_URL     — Ollama endpoint (may be behind auth proxy)
    OLLAMA_AI_MODEL    — model name (default: gemma4:26b)
    OLLAMA_USERNAME    — HTTP Basic Auth username (optional)
    OLLAMA_PASSWORD    — HTTP Basic Auth password (optional)

Usage:
    uv run examples/25_customer_enrichment.py
"""

from __future__ import annotations

import os
from pydantic import BaseModel, Field
from typing import Literal

import quartermaster_sdk as qm
from quartermaster_tools import tool
from quartermaster_tools.builtin.web_search.scraper import web_scraper
from quartermaster_tools.builtin.web_search.duckduckgo import duckduckgo_search


# ── Schema ────────────────────────────────────────────────────────────

class CustomerProfile(BaseModel):
    """Structured customer profile extracted from web research."""

    industry: str | None = Field(
        default=None,
        description="What industry/sector (e.g. office supplies, food production, logistics)",
    )
    company_size: Literal["micro", "small", "medium", "large", "enterprise"] | None = Field(
        default=None,
        description="Estimated company size",
    )
    customer_type: Literal[
        "manufacturer", "retailer", "reseller", "distributor", "end-user"
    ] | None = Field(default=None, description="Business type")
    contact_person: str | None = Field(
        default=None,
        description="Name of owner, director, or key contact person",
    )
    email: str | None = Field(
        default=None,
        description="Company email address (info@, office@, or personal)",
    )
    phone: str | None = Field(
        default=None,
        description="Phone number with country code",
    )
    address: str | None = Field(default=None, description="Street address")
    city: str | None = Field(default=None, description="City name")
    postal_code: str | None = Field(default=None, description="Postal/ZIP code")
    country: str | None = Field(default=None, description="Country name")
    website: str | None = Field(
        default=None, description="Full URL including https://"
    )
    vat_id: str | None = Field(
        default=None, description="VAT number / tax ID / IČO / DIČ"
    )
    notes: str | None = Field(
        default=None,
        description="2-3 sentences describing the company",
    )


# ── Tools ─────────────────────────────────────────────────────────────


@tool()
def scrape_url(url: str) -> dict:
    """Fetch a web page and return its text content.

    Args:
        url: The URL to scrape (must start with http:// or https://).
    """
    ctx = qm.current_context()
    if ctx is not None:
        ctx.emit_progress(f"Scraping {url}...", percent=0.5)
    result = web_scraper(url=url, output_format="text", timeout=15)
    if ctx is not None:
        text = result.get("content", "") if isinstance(result, dict) else str(result)
        ctx.emit_progress(
            f"Scraped {len(text)} chars from {url}",
            percent=1.0,
            chars=len(text),
        )
    return result


@tool()
def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
    """
    ctx = qm.current_context()
    if ctx is not None:
        ctx.emit_progress(f"Searching: {query}")
    result = duckduckgo_search(query=query, max_results=max_results)
    if ctx is not None:
        count = len(result.get("results", [])) if isinstance(result, dict) else 0
        ctx.emit_custom("search_done", {"query": query, "result_count": count})
    return result


# ── Sub-graph: scrape a known URL and extract info ────────────────────

def build_web_scrape_graph(url: str, site_name: str) -> qm.GraphBuilder:
    """Build a sub-graph that scrapes a specific URL and extracts info.

    The sub-graph:
      1. Static node injects the URL
      2. Agent calls scrape_url to fetch the page
      3. Instruction node extracts relevant business info from the content

    This is composed into the main graph via .use() — the sub-graph's
    nodes are inlined at build time.
    """
    return (
        qm.Graph(f"Scrape {site_name}")
        .static(f"{site_name} URL", text=url)
        .agent(
            f"Fetch {site_name}",
            tools=[scrape_url],
            max_iterations=2,
            # model + provider inherited from qm.configure() — no need
            # to repeat them on every node.
            system_instruction=(
                f"You are a web scraper. Fetch the content from this URL: {url}\n"
                "Use the scrape_url tool to get the page content. "
                "Return a brief summary of what you found — focus on "
                "company contact info, address, phone, email, VAT ID, "
                "and what the company does."
            ),
            capture_as=f"scrape_{site_name.lower().replace(' ', '_')}",
        )
    )


# ── Main graph ────────────────────────────────────────────────────────

def build_enrichment_graph(
    known_urls: list[tuple[str, str]],
) -> qm.GraphBuilder:
    """Build the full enrichment pipeline.

    Args:
        known_urls: List of (url, site_name) tuples for static scraping.
                    Example: [("https://bizi.si/COMPANY", "Bizi.si")]
    """
    graph = qm.Graph("Customer Enrichment")

    # Step 1: User provides company name + country
    graph.user("Enter company name and country")

    # Step 2: Static web scrapes (known URLs)
    for url, site_name in known_urls:
        sub = build_web_scrape_graph(url, site_name)
        graph.use(sub)

    # Step 3: Dynamic research agent — searches for more info
    graph.agent(
        "Research Agent",
        tools=[search_web, scrape_url],
        max_iterations=8,
        capture_as="research",
        system_instruction=(
            "You are a business research agent. Based on the company name "
            "and any information gathered from the static web scrapes above, "
            "search for additional information about this company.\n\n"
            "Use search_web to find the company on:\n"
            "- Their official website (look for contact page, about page, imprint)\n"
            "- Business registries (for VAT ID, registration number)\n"
            "- LinkedIn or social media profiles\n"
            "- Google Maps (for address verification)\n\n"
            "Use scrape_url to read promising search results.\n\n"
            "Focus on finding: email, phone, address, VAT ID, contact person, "
            "company size, industry, and what they do.\n\n"
            "Compile ALL findings into a structured summary. Cite sources."
        ),
    )

    # Step 4: Summarise all findings into a single research document.
    # The structured extraction happens AFTER the graph run via
    # qm.instruction_form() — a separate one-shot call that gets the
    # full research notes as input and returns a validated Pydantic model.
    graph.instruction(
        "Compile Notes",
        capture_as="notes",
        system_instruction=(
            "You are a research compiler. Combine ALL information from the "
            "static web scrapes and the dynamic research above into a single, "
            "structured research document. Include every piece of contact "
            "info you found: email, phone, address, VAT ID, website, "
            "contact person, industry, company size. Cite the source for "
            "each fact. This document will be parsed by a downstream "
            "extractor — be thorough and precise."
        ),
    )

    return graph


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Configure Ollama from .env ────────────────────────────────
    # Reads: OLLAMA_API_URL, OLLAMA_AI_MODEL, OLLAMA_USERNAME, OLLAMA_PASSWORD
    ollama_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/v1")
    ollama_model = os.environ.get("OLLAMA_AI_MODEL", "gemma4:26b")
    ollama_user = os.environ.get("OLLAMA_USERNAME")
    ollama_pass = os.environ.get("OLLAMA_PASSWORD")

    auth = (ollama_user, ollama_pass) if ollama_user and ollama_pass else None

    qm.configure(
        provider="ollama",
        base_url=ollama_url,
        default_model=ollama_model,
        auth=auth,
        timeout=120,
    )

    # Example: enrich a Slovenian company with known registry URLs
    company_name = "Kolibri d.o.o."
    country = "Slovenia"

    # Known URLs where we expect to find info
    known_urls = [
        ("https://www.bizi.si/KOLIBRI-D-O-O/", "Bizi.si"),
        ("https://www.google.com/maps/search/Kolibri+d.o.o.+Slovenia", "Google Maps"),
    ]

    print("=" * 60)
    print("  Customer Enrichment Pipeline")
    print(f"  Provider: ollama @ {ollama_url}")
    print(f"  Model: {ollama_model}")
    print(f"  Auth: {'basic' if auth else 'none'}")
    print(f"  Company: {company_name}")
    print(f"  Country: {country}")
    print(f"  Static sources: {len(known_urls)}")
    print("=" * 60)
    print()

    graph = build_enrichment_graph(known_urls)

    # Run the graph — collects research from static + dynamic sources
    print("--- Running enrichment ---\n")
    result = qm.run(graph, f"{company_name}, {country}")

    # The graph produced research notes; now extract structured profile
    # via a separate instruction_form call (Pydantic-validated output).
    research_notes = result["notes"].output_text or result.text
    print("\n--- Extracting structured profile ---\n")
    profile: CustomerProfile = qm.instruction_form(
        schema=CustomerProfile,
        system=(
            "You are a data extraction specialist. Based on the research "
            "notes below, extract every available field into the schema.\n\n"
            "Rules:\n"
            "- Only include information you're confident about\n"
            "- Leave fields as null when not found (don't guess)\n"
            "- For VAT ID, use the format from the company's country "
            "(SI######## for Slovenia, DE######### for Germany, etc.)\n"
            "- For phone numbers, include the country code\n"
            "- For website, include the full URL with https://\n"
            "- The notes field should be 2-3 sentences describing the company"
        ),
        user=research_notes,
    )

    # Show the extracted profile
    print("=" * 60)
    print("  EXTRACTED PROFILE")
    print("=" * 60)
    for key, value in profile.model_dump().items():
        if value is not None:
            print(f"  {key:20s}: {value}")

    # Show research trace
    print(f"\n--- Trace ---")
    print(f"  Duration: {result.trace.duration_seconds:.1f}s")
    print(f"  Tool calls: {len(result.trace.tool_calls)}")
    for tc in result.trace.tool_calls:
        print(f"    {tc.get('tool', '?')}({', '.join(f'{k}={v!r}' for k, v in tc.get('arguments', {}).items())})")

    # Save trace for regression testing
    print(f"\n--- JSONL trace ({len(result.trace.events)} events) ---")
    print(result.trace.as_jsonl()[:500] + "...")

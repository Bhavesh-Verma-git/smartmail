"""
src/research/tavily_provider.py
-------------------------------
Tavily Search API implementation of ResearchProvider.

Key requirements:
1. Optional provider: If API key is missing or search fails, seamlessly falls back to HTTP fetching.
2. Normalized domain caching: Avoid repeating identical searches within the session or batch.
3. Provenance & source URLs retained: Records actual web page source URLs returned by Tavily.
4. Bounded retries & rate-limit handling: Handles HTTP 429 with exponential backoff (max 2 retries).
5. Untrusted snippets: Search snippets are tagged as source_type="tavily_search" and do not
   count as conclusive proof on their own without domain/site corroboration.
6. Privacy: Never sends recipient email addresses to Tavily.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from src.config import get_tavily_api_key, get_tavily_timeout
from src.research.models import (
    EvidenceItem,
    RETRIEVAL_BLOCKED,
    RETRIEVAL_ERROR,
    RETRIEVAL_NETWORK_ERROR,
    RETRIEVAL_OK,
    RETRIEVAL_SKIPPED,
)
from src.research.providers import ResearchProvider

logger = logging.getLogger(__name__)

TAVILY_API_ENDPOINT = "https://api.tavily.com/search"


class TavilySearchProvider(ResearchProvider):
    """
    Search provider that queries Tavily API for company official websites and AI/ML evidence.
    Includes in-memory caching by query/domain and retry/backoff for rate limits.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
    ) -> None:
        self._api_key = api_key.strip() if api_key is not None else get_tavily_api_key()
        self._timeout = timeout or get_tavily_timeout()
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        # In-memory query cache: query -> list[EvidenceItem]
        self._cache: dict[str, list[EvidenceItem]] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def search_company(
        self,
        company_name: str,
        domain: str = "",
        max_results: int = 5,
    ) -> list[EvidenceItem]:
        """
        Search for information about a company using its name and/or domain.
        NEVER sends recipient personal emails to Tavily!
        """
        if not self.is_configured:
            return []

        # Construct safe query using only company/domain info
        clean_company = (company_name or "").strip()
        clean_domain = (domain or "").strip()

        if clean_company and clean_domain:
            query = f'"{clean_company}" site:{clean_domain} OR "{clean_company}" AI ML technology'
        elif clean_domain:
            query = f'site:{clean_domain} OR "{clean_domain}" AI technology'
        elif clean_company:
            query = f'"{clean_company}" AI machine learning engineering'
        else:
            return []

        cache_key = query.lower().strip()
        if cache_key in self._cache:
            return [
                EvidenceItem(
                    source_url=it.source_url,
                    source_type=it.source_type,
                    title=it.title,
                    extracted_text=it.extracted_text,
                    retrieved_at=EvidenceItem.now_utc(),
                    retrieval_status=it.retrieval_status,
                    retrieval_note=f"{it.retrieval_note} (cached)",
                    supports=list(it.supports),
                    limitations=list(it.limitations),
                    claim_supported=it.claim_supported,
                )
                for it in self._cache[cache_key]
            ]

        # Call Tavily REST endpoint with bounded retries
        evidence_items = self._execute_search(query, max_results=max_results)
        self._cache[cache_key] = evidence_items
        return evidence_items

    def _execute_search(self, query: str, max_results: int = 5) -> list[EvidenceItem]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
        }

        retries = 0
        backoff = self._initial_backoff

        while retries <= self._max_retries:
            try:
                response = requests.post(
                    TAVILY_API_ENDPOINT,
                    json=payload,
                    timeout=self._timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    items: list[EvidenceItem] = []
                    for r in results:
                        url = r.get("url", "")
                        title = r.get("title", "")
                        snippet = r.get("content", "")
                        items.append(
                            EvidenceItem(
                                source_url=url,
                                source_type="tavily_search",
                                title=title,
                                extracted_text=snippet[:2000],
                                retrieved_at=EvidenceItem.now_utc(),
                                retrieval_status=RETRIEVAL_OK,
                                retrieval_note="Tavily Web Search API result",
                                supports=["company_identity", "technical_relevance"],
                                limitations=["Search snippet only; subject to primary site verification."],
                                claim_supported=f"Tavily snippet from {url}",
                            )
                        )
                    return items

                elif response.status_code == 429:
                    retries += 1
                    if retries > self._max_retries:
                        logger.warning("Tavily API rate limit exceeded after max retries.")
                        return [
                            EvidenceItem(
                                source_url=TAVILY_API_ENDPOINT,
                                source_type="tavily_search",
                                title="Tavily Rate Limited",
                                extracted_text="",
                                retrieved_at=EvidenceItem.now_utc(),
                                retrieval_status=RETRIEVAL_ERROR,
                                retrieval_note="HTTP 429: Rate limit reached",
                                limitations=["Tavily quota or rate limit exceeded; falling back to direct HTTP."],
                            )
                        ]
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                elif response.status_code in (401, 403):
                    logger.warning("Tavily API authentication failed. Check TAVILY_API_KEY.")
                    return [
                        EvidenceItem(
                            source_url=TAVILY_API_ENDPOINT,
                            source_type="tavily_search",
                            title="Tavily Auth Error",
                            extracted_text="",
                            retrieved_at=EvidenceItem.now_utc(),
                            retrieval_status=RETRIEVAL_BLOCKED,
                            retrieval_note=f"HTTP {response.status_code}: Auth failed",
                            limitations=["Tavily API key is invalid or unauthorized."],
                        )
                    ]
                else:
                    return [
                        EvidenceItem(
                            source_url=TAVILY_API_ENDPOINT,
                            source_type="tavily_search",
                            title=f"Tavily Error HTTP {response.status_code}",
                            extracted_text="",
                            retrieved_at=EvidenceItem.now_utc(),
                            retrieval_status=RETRIEVAL_ERROR,
                            retrieval_note=f"HTTP {response.status_code}",
                            limitations=[f"Tavily request failed with status {response.status_code}."],
                        )
                    ]

            except requests.exceptions.Timeout:
                retries += 1
                if retries > self._max_retries:
                    return [
                        EvidenceItem(
                            source_url=TAVILY_API_ENDPOINT,
                            source_type="tavily_search",
                            title="Tavily Timeout",
                            extracted_text="",
                            retrieved_at=EvidenceItem.now_utc(),
                            retrieval_status=RETRIEVAL_NETWORK_ERROR,
                            retrieval_note="Connection timed out",
                            limitations=[f"Tavily request timed out after {self._timeout}s."],
                        )
                    ]
                time.sleep(backoff)
                backoff *= 2

            except Exception as exc:
                return [
                    EvidenceItem(
                        source_url=TAVILY_API_ENDPOINT,
                        source_type="tavily_search",
                        title="Tavily Request Exception",
                        extracted_text="",
                        retrieved_at=EvidenceItem.now_utc(),
                        retrieval_status=RETRIEVAL_ERROR,
                        retrieval_note=f"{type(exc).__name__}",
                        limitations=[f"Unexpected error communicating with Tavily: {type(exc).__name__}."],
                    )
                ]

        return []

    def fetch(
        self,
        url: str,
        source_type: str = "web_page",
        supports: list[str] | None = None,
    ) -> EvidenceItem:
        """
        Direct URL fetching delegates to HttpPageProvider, reserving Tavily API
        for semantic search, company discovery, and intelligence.
        """
        if not url:
            return EvidenceItem(
                source_url="",
                source_type=source_type,
                retrieved_at=EvidenceItem.now_utc(),
                retrieval_status=RETRIEVAL_SKIPPED,
                retrieval_note="No URL supplied",
            )
        from src.research.providers import HttpPageProvider
        http_fetcher = HttpPageProvider(timeout_seconds=self._timeout)
        return http_fetcher.fetch(url=url, source_type=source_type, supports=supports)


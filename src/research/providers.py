"""
src/research/providers.py
--------------------------
Research provider interface and HTTP implementation.

## Why an abstract base class?

The rest of the research module calls provider.fetch(url) and gets back
a PageResult.  In production, HttpPageProvider makes real HTTP requests.
In tests, a MockProvider returns pre-canned PageResult objects without
touching the network.  Swapping the provider requires no changes to the
check functions or the top-level researcher.

## HttpPageProvider rules

- Uses requests with a configurable timeout (default 10 s).
- Follows up to 5 redirects.
- Sends a plain browser-like User-Agent so well-behaved sites serve HTML.
- Treats any HTTP error status (4xx, 5xx) as a retrieval error; records
  the status code in retrieval_note.
- Extracts visible text via BeautifulSoup with the stdlib html.parser.
  No binary extensions required on Windows.
- Caps extracted_text at MAX_TEXT_CHARS (2 000) to keep results compact.
- Does NOT follow JavaScript redirects or render dynamic content.
- Does NOT scrape pages behind authentication.
- Treats all retrieved content as untrusted text; never executes it.
- Respects the caller-supplied timeout; ConnectionError and Timeout
  are caught and reported as RETRIEVAL_ERROR.

## Free-tier note

HttpPageProvider requires no API key and no account.
It only fetches URLs you explicitly supply (website, role_or_job_url,
source_url fields from the ContactRecord).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from src.research.models import (
    EvidenceItem,
    RETRIEVAL_OK,
    RETRIEVAL_ERROR,
    RETRIEVAL_SKIPPED,
    RETRIEVAL_EMPTY,
    RETRIEVAL_BLOCKED,
    RETRIEVAL_JS_ONLY,
    RETRIEVAL_NETWORK_ERROR,
)

# Maximum characters of page text to keep
MAX_TEXT_CHARS = 2_000

# A short browser-like User-Agent so sites return real HTML rather than a
# bot-detection page. We identify ourselves honestly as a research tool.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; SmartMailResearch/1.0; "
    "+https://github.com/user/smartmail)"
)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ResearchProvider(ABC):
    """
    Abstract interface for fetching a URL and returning an EvidenceItem.

    Subclass this to add new providers (e.g. a search API adapter).
    All implementations must:
      - Return an EvidenceItem with retrieved_at set to a UTC ISO timestamp.
      - Never raise — capture errors inside the EvidenceItem.
      - Never invent content; extracted_text must reflect what was retrieved.
    """

    @abstractmethod
    def fetch(
        self,
        url: str,
        source_type: str,
        supports: list[str],
    ) -> EvidenceItem:
        """
        Fetch the given URL and return a populated EvidenceItem.
        """


# ---------------------------------------------------------------------------
# HTTP implementation
# ---------------------------------------------------------------------------

class HttpPageProvider(ResearchProvider):
    """
    Fetches publicly accessible web pages using requests + BeautifulSoup.

    Parameters
    ----------
    timeout_seconds : int
        Maximum seconds to wait for a response. Applies to both the
        connection and the read phase.
    """

    def __init__(self, timeout_seconds: int = 10) -> None:
        self._timeout = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._session.max_redirects = 5

    def fetch(
        self,
        url: str,
        source_type: str,
        supports: list[str],
    ) -> EvidenceItem:
        """
        Fetch *url* and return an EvidenceItem with precise retrieval status.
        """
        item = EvidenceItem(
            source_url=url,
            source_type=source_type,
            retrieved_at=EvidenceItem.now_utc(),
            supports=list(supports),
        )

        try:
            response = self._session.get(url, timeout=self._timeout)
            item.retrieval_note = f"HTTP {response.status_code}"

            # 1. Check for access blocks
            if response.status_code in (401, 403, 429):
                item.retrieval_status = RETRIEVAL_BLOCKED
                item.retrieval_note = f"HTTP {response.status_code}"
                item.claim_supported = f"Access blocked (HTTP {response.status_code})"
                item.limitations.append(
                    f"Access blocked (HTTP {response.status_code}) for {url}."
                )
                return item

            if not response.ok:
                item.retrieval_status = RETRIEVAL_ERROR
                item.retrieval_note = f"HTTP {response.status_code}"
                item.claim_supported = f"Server returned HTTP {response.status_code}"
                item.limitations.append(
                    f"Server returned HTTP {response.status_code} for {url}."
                )
                return item

            # 2. Extract content
            content_type = response.headers.get("content-type", "")
            raw_html = response.text or ""

            is_html = not content_type or "html" in content_type.lower() or "<html" in raw_html.lower() or not raw_html.strip()

            if not is_html:
                item.retrieval_status = RETRIEVAL_OK
                item.title = f"Non-HTML response ({content_type})"
                item.extracted_text = raw_html[:MAX_TEXT_CHARS]
                item.claim_supported = f"Non-HTML resource ({content_type}) retrieved"
                return item

            soup = BeautifulSoup(raw_html, "html.parser")
            title_tag = soup.find("title")
            item.title = title_tag.get_text(strip=True) if title_tag else ""

            # Check for client-side JavaScript requirement
            is_spa = bool(
                soup.find(id=re.compile(r"^(root|app|__next)$", re.IGNORECASE))
                or "javascript to run this app" in raw_html.lower()
                or "enable javascript" in raw_html.lower()
            )

            # Decompose decorative/script tags
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()

            raw_text = soup.get_text(separator=" ")
            clean_text = re.sub(r"\s+", " ", raw_text).strip()

            if not clean_text:
                if is_spa:
                    item.retrieval_status = RETRIEVAL_JS_ONLY
                    item.claim_supported = "Requires JavaScript runtime (SPA)"
                    item.limitations.append(
                        f"Page at {url} requires client-side JavaScript execution."
                    )
                else:
                    item.retrieval_status = RETRIEVAL_EMPTY
                    item.claim_supported = "Empty page content"
                    item.limitations.append(
                        f"Retrieved page from {url} contained no visible text content."
                    )
                return item

            item.extracted_text = clean_text[:MAX_TEXT_CHARS]
            item.retrieval_status = RETRIEVAL_OK
            item.claim_supported = f"Retrieved {len(clean_text)} characters from {source_type}"

        except requests.exceptions.Timeout:
            item.retrieval_status = RETRIEVAL_NETWORK_ERROR
            item.retrieval_note = "Timeout"
            item.claim_supported = "Connection timed out"
            item.limitations.append(f"Request to {url} timed out after {self._timeout} s.")
        except requests.exceptions.TooManyRedirects:
            item.retrieval_status = RETRIEVAL_ERROR
            item.retrieval_note = "TooManyRedirects"
            item.claim_supported = "Too many redirects"
            item.limitations.append(f"{url} returned too many redirects.")
        except requests.exceptions.ConnectionError as exc:
            item.retrieval_status = RETRIEVAL_NETWORK_ERROR
            item.retrieval_note = "ConnectionError"
            item.claim_supported = f"Connection error: {type(exc).__name__}"
            item.limitations.append(f"Could not connect to {url}: {type(exc).__name__}.")
        except Exception as exc:
            item.retrieval_status = RETRIEVAL_ERROR
            item.retrieval_note = f"Error: {type(exc).__name__}"
            item.claim_supported = f"Error fetching {url}: {type(exc).__name__}"
            item.limitations.append(f"Unexpected error fetching {url}: {type(exc).__name__}.")

        return item

    def fetch_page(
        self,
        url: str,
        source_type: str = "web_page",
        supports: list[str] | None = None,
    ) -> EvidenceItem:
        """Convenience wrapper around fetch()."""
        return self.fetch(url=url, source_type=source_type, supports=supports or [])


def get_default_research_provider() -> ResearchProvider:
    """
    Factory function: returns TavilySearchProvider if TAVILY_API_KEY is configured,
    otherwise returns the default HttpPageProvider.
    """
    from src.config import get_tavily_api_key
    from src.research.tavily_provider import TavilySearchProvider

    if get_tavily_api_key():
        return TavilySearchProvider()
    return HttpPageProvider()

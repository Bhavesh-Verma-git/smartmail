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

- Uses requests with a configurable timeout (default (3.0 s connect, 5.0 s read)).
- Follows up to 5 redirects; the citation URL is the FINAL URL after redirects.
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
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
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
# URL safety

def _blocked_url_reason(url: str) -> str | None:
    """Return a block reason for an unsafe fetch URL, else None.

    Blocks non-http(s) schemes, URLs with embedded credentials, and
    literal private/loopback/link-local/reserved/multicast addresses.
    Hostnames are resolved when possible; a resolution failure is not
    treated as a block (the fetch will fail on its own).
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except Exception:
        return "unparseable URL"
    if parsed.scheme not in ("http", "https"):
        return f"unsupported scheme '{parsed.scheme}'"
    if parsed.username or parsed.password:
        return "URL contains embedded credentials"
    host = (parsed.hostname or "").strip()
    if not host:
        return "missing host"
    try:
        ip = ipaddress.ip_address(host)
        candidates = [ip]
    except ValueError:
        candidates = []
        try:
            for _fam, _typ, _proto, _canon, sockaddr in socket.getaddrinfo(host, None):
                try:
                    candidates.append(ipaddress.ip_address(sockaddr[0]))
                except ValueError:
                    continue
        except Exception:
            return None
    # Block loopback, RFC1918 private nets, link-local, unspecified,
    # multicast and reserved addresses. Note: ipaddress.is_private is
    # deliberately NOT used -- it also flags benchmarking/CGNAT ranges
    # (e.g. 198.18/15) that are not internal service targets, and some
    # test/sandbox resolvers synthesise those for public names.
    _rfc1918 = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    for ip in candidates:
        if (ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified
                or any(ip in net for net in _rfc1918)):
            return f"host resolves to non-public address {ip}"
    return None


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

    def __init__(self, timeout_seconds: int | float | tuple[float, float] = (3.0, 5.0), max_retries: int = 0) -> None:
        if isinstance(timeout_seconds, (int, float)):
            self._timeout: tuple[float, float] = (min(3.0, float(timeout_seconds)), float(timeout_seconds))
        else:
            self._timeout = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._session.max_redirects = 5
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.2,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

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

        # Strict URL validation (SSRF guard for localhost deployment):
        # only http/https, no embedded credentials, and no literal or
        # resolved private/loopback/link-local/reserved address. A DNS
        # failure does not block here -- the request itself will then
        # fail and be recorded as a retrieval error.
        blocked = _blocked_url_reason(url)
        if blocked:
            item.retrieval_status = RETRIEVAL_BLOCKED
            item.retrieval_note = "Blocked URL"
            item.claim_supported = f"URL blocked: {blocked}"
            item.limitations.append(f"Fetch blocked for {url}: {blocked}.")
            return item

        try:
            response = self._session.get(url, timeout=self._timeout)
            item.retrieval_note = f"HTTP {response.status_code}"
            # Cite the final URL after redirects, not the requested one:
            # citing the requested URL when content came from elsewhere
            # misattributes the evidence.
            final_url = getattr(response, "url", "") or url
            if final_url != url:
                item.source_url = final_url
                item.retrieval_note += f" (redirected to {final_url})"

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
                # A non-HTML response (PDF, JSON, binary) is NOT page
                # evidence: its raw bytes must not be counted as retrieved
                # company text. Record it as an error so downstream checks
                # never treat it as RETRIEVAL_OK evidence.
                item.retrieval_status = RETRIEVAL_ERROR
                item.title = f"Non-HTML response ({content_type})"
                item.extracted_text = ""
                item.claim_supported = f"Non-HTML resource ({content_type}); not usable as page evidence"
                item.limitations.append(
                    f"Response from {item.source_url} was non-HTML ({content_type}); "
                    "its content was not used as evidence."
                )
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

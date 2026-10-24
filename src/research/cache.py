"""
src/research/cache.py
---------------------
Thread-safe company research and page fetch caching.

Requirements addressed:
1. Thread-safe in-memory cache with expiration/freshness checking.
2. Avoids duplicate HTTP requests when multiple contacts share a company, domain, or website.
3. Only caches valid, non-error retrievals to ensure transient network failures can retry.
4. Thread-safe copying of EvidenceItem records to prevent cross-thread mutation.
"""
from __future__ import annotations

import copy
import dataclasses
import threading
import time
from typing import Any

from src.research.models import (
    EvidenceItem,
    RETRIEVAL_OK,
)
from src.research.providers import ResearchProvider


class ResearchCache:
    """
    Thread-safe in-memory cache for fetched EvidenceItem objects and company research.
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._lock = threading.RLock()
        self._page_cache: dict[str, tuple[float, EvidenceItem]] = {}
        self._company_cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get_page(self, url: str) -> EvidenceItem | None:
        """Retrieve a cached page EvidenceItem if present and fresh."""
        if not url:
            return None
        norm_url = url.strip().lower()
        with self._lock:
            if norm_url in self._page_cache:
                timestamp, item = self._page_cache[norm_url]
                if time.time() - timestamp <= self._ttl:
                    self.hits += 1
                    # Return an isolated copy so consumers cannot mutate cached state
                    return dataclasses.replace(
                        item,
                        supports=list(item.supports),
                        limitations=list(item.limitations),
                        contradictions=list(item.contradictions),
                    )
                else:
                    del self._page_cache[norm_url]
            self.misses += 1
            return None

    def put_page(self, url: str, item: EvidenceItem) -> None:
        """
        Store an EvidenceItem in cache.
        Only caches if retrieval was successful (RETRIEVAL_OK) to prevent
        caching transient timeouts or network blips.
        """
        if not url or item.retrieval_status != RETRIEVAL_OK:
            return
        norm_url = url.strip().lower()
        with self._lock:
            # Store isolated copy
            stored_item = dataclasses.replace(
                item,
                supports=list(item.supports),
                limitations=list(item.limitations),
                contradictions=list(item.contradictions),
            )
            self._page_cache[norm_url] = (time.time(), stored_item)

    def get_company_data(self, key: str) -> Any | None:
        """Retrieve cached company research result if fresh."""
        if not key:
            return None
        norm_key = key.strip().lower()
        with self._lock:
            if norm_key in self._company_cache:
                timestamp, data = self._company_cache[norm_key]
                if time.time() - timestamp <= self._ttl:
                    return copy.deepcopy(data)
                else:
                    del self._company_cache[norm_key]
            return None

    def put_company_data(self, key: str, data: Any) -> None:
        """Store company research result."""
        if not key or data is None:
            return
        norm_key = key.strip().lower()
        with self._lock:
            self._company_cache[norm_key] = (time.time(), copy.deepcopy(data))

    def clear(self) -> None:
        """Clear cache contents."""
        with self._lock:
            self._page_cache.clear()
            self._company_cache.clear()
            self.hits = 0
            self.misses = 0


class CachedResearchProvider(ResearchProvider):
    """
    Thread-safe decorator around any ResearchProvider that caches valid EvidenceItems.
    """

    def __init__(
        self,
        provider: ResearchProvider,
        cache: ResearchCache | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache if cache is not None else ResearchCache()

    def fetch(
        self,
        url: str,
        source_type: str,
        supports: list[str],
    ) -> EvidenceItem:
        """Fetch URL with cache check and thread safety."""
        # Check cache first
        cached = self.cache.get_page(url)
        if cached is not None:
            # Merge requested supports into the returned copy
            merged_supports = list(set(cached.supports + supports))
            return dataclasses.replace(
                cached,
                source_type=source_type or cached.source_type,
                supports=merged_supports,
            )

        # Cache miss: fetch from inner provider
        item = self.provider.fetch(url, source_type, supports)
        if item.retrieval_status == RETRIEVAL_OK:
            self.cache.put_page(url, item)
        return item

    def research(self, inp: Any) -> Any:
        """Delegate research call if underlying provider supports it, otherwise run researcher."""
        if hasattr(self.provider, "research"):
            return self.provider.research(inp)
        from src.research.researcher import research_contact
        return research_contact(inp, provider=self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

"""Rate limiting utilities for Wikidata API requests."""

import asyncio
import logging
from threading import Semaphore
from typing import Dict, Optional

from wemg.agents.tools.wikidata.constants import MAX_CONCURRENT_REQUESTS

logger = logging.getLogger(__name__)

# Global semaphore for rate limiting (sync)
_wikidata_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# Async semaphores per event loop (thread-safe dict)
_async_semaphores: Dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
_async_semaphore_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None


def get_sync_semaphore() -> Semaphore:
    """Get the global sync semaphore for rate limiting."""
    return _wikidata_semaphore


def get_async_semaphore() -> asyncio.Semaphore:
    """Get or create async semaphore for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, create a new semaphore (will be associated with loop when used)
        return asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Use loop as key to ensure one semaphore per event loop
    if loop not in _async_semaphores:
        _async_semaphores[loop] = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    return _async_semaphores[loop]


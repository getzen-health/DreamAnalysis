"""In-memory rate limiter for the ML backend.

Uses a sliding window counter per IP address. No external dependencies required.
Thread-safe via a lock around the counter dict.

Usage in FastAPI middleware:
    limiter = RateLimiter(max_requests=100, window_seconds=60)
    if not limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, ...)
"""

import threading
import time
from collections import defaultdict
from typing import Dict, List


class RateLimiter:
    """Sliding-window rate limiter keyed by client identifier (typically IP).

    Args:
        max_requests: Maximum number of requests allowed per window.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id is allowed.

        Returns True if the request is within the rate limit, False otherwise.
        Also prunes expired timestamps to keep memory bounded.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            timestamps = self._requests[client_id]

            # Prune expired entries
            self._requests[client_id] = [t for t in timestamps if t > cutoff]
            timestamps = self._requests[client_id]

            if len(timestamps) >= self.max_requests:
                return False

            timestamps.append(now)
            return True

    def reset(self, client_id: str) -> None:
        """Clear rate limit state for a specific client."""
        with self._lock:
            self._requests.pop(client_id, None)

    def reset_all(self) -> None:
        """Clear all rate limit state."""
        with self._lock:
            self._requests.clear()

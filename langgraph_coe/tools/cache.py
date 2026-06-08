from __future__ import annotations

import json
from typing import Any, Mapping, Optional


class RedisDictCache:
    """Tiny JSON cache wrapper with prefix-based default TTLs."""

    def __init__(self, *, client: Any, ttls: Optional[Mapping[str, int]] = None):
        self._client = client
        self._ttls = dict(ttls or {})

    def _default_ttl_for_key(self, key: str) -> Optional[int]:
        first, _, rest = key.partition(":")
        if first == "wd" and rest:
            second = rest.split(":", 1)[0]
            if second in self._ttls:
                return self._ttls[second]
        return self._ttls.get(first)

    def get(self, key: str) -> Any:
        try:
            raw = self._client.get(key)
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception:
                return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any, *, ex: Optional[int] = None) -> None:
        ttl = ex if ex is not None else self._default_ttl_for_key(key)
        try:
            payload = json.dumps(value)
        except Exception:
            return

        try:
            if ttl is None:
                self._client.set(key, payload)
            else:
                self._client.set(key, payload, ex=ttl)
        except Exception:
            return

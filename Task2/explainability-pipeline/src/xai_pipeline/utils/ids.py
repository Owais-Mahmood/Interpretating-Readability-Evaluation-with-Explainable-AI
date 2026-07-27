from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_id(payload: Any, length: int = 16) -> str:
    serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:length]

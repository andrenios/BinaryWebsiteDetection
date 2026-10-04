"""Per-site stage timing written to results/timing/<experiment>.csv
(ground rule 12). One row per (site, variant, stage)."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

FIELDS = ["ts", "experiment", "site_id", "variant", "stage", "seconds", "note"]


class TimingLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        new = not self.path.exists()
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        if new:
            self._w.writeheader()

    def record(self, experiment: str, site_id: str, variant: str, stage: str,
               seconds: float, note: str = "") -> None:
        with self._lock:
            self._w.writerow({"ts": round(time.time(), 3), "experiment": experiment,
                              "site_id": site_id, "variant": variant, "stage": stage,
                              "seconds": round(seconds, 6), "note": note})
            self._f.flush()

    def close(self) -> None:
        self._f.close()

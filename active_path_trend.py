#!/usr/bin/env python3
"""
Plot how many Scheduled tuples are active over time and
flag moments when only one path is scheduled.
"""

import re
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path

LOG_FILE = Path("/home/paul/data_quiche/logs_backup/success_server_per_packet/quiche_server.log")
OUT_PNG  = LOG_FILE.with_suffix(".scheduled_tuples.png")

# ── regex for the timestamp + Scheduled tuples list ────────────────
#   [2025-04-16T14:05:50.591471372Z DEBUG quiche_server] ... Scheduled tuples:[ ... ]
pat = re.compile(
    r"""\[
        (?P<ts>[\d\-T:\.]+)   # ISO‑like timestamp up to the dot
        Z .*?                # 'Z' and anything up to 'Scheduled tuples:'
        Scheduled\ tuples:\[ (?P<list>[^\]]*) \]""",
    re.VERBOSE,
)

ts_list, counts, single_flags = [], [], []

with LOG_FILE.open() as fh:
    for line in fh:
        m = pat.search(line)
        if not m:
            continue

        raw_ts = m.group("ts")              # e.g. 2025-04-16T14:05:50.591471372
        if "." in raw_ts:
            left, frac = raw_ts.split(".")
            frac = (frac + "000000")[:6]    # pad/trim → 6‑digit microseconds
            fixed_ts = f"{left}.{frac}"
        else:
            fixed_ts = raw_ts

        ts = dt.datetime.strptime(fixed_ts, "%Y-%m-%dT%H:%M:%S.%f")

        tuples_raw = m.group("list")
        n_tuples   = len(re.findall(r"\([^)]*\)", tuples_raw))

        ts_list.append(ts)
        counts.append(n_tuples)
        single_flags.append(n_tuples == 1)

# ── plotting ───────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(ts_list, counts, marker="o", label="Scheduled tuple count")

# vertical dashed red line when only one path is active
for t, single in zip(ts_list, single_flags):
    if single:
        plt.axvline(t, linestyle="--", alpha=0.35)

plt.title("Scheduled tuples over time")
plt.xlabel("Time")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.savefig(OUT_PNG, dpi=150)
plt.show()
print(f"Plot saved → {OUT_PNG}")

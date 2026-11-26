"""Where a decode step's wall time goes that no kernel accounts for: the GPU's idle gaps in
an nsys profile (bench/profile_decode.sh), per engine step. No GPU needed.

    venv/bin/python bench/gpu_idle.py .local/prof/<name>.nsys-rep [marker]

The marker is a kernel that runs once per step (default `_suffix_lookup_kernel`, the
context lookup); steps are the spans between its launches, the middle half of them is
reported: the period, the kernel time in it, the idle time, and what the GPU
waited between (the kernel before and after each gap, summed).
"""

import collections
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path


def _events(path):
    if path.suffix != ".sqlite":
        out = path.with_suffix(".sqlite")
        if not out.exists():
            cmd = ["nsys", "export", "--type", "sqlite", "-o", str(out), str(path)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        path = out
    db = sqlite3.connect(path)
    cols = [r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    name = "shortName" if "shortName" in cols else "demangledName"
    events = db.execute(
        f"select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        f"join StringIds s on s.id = k.{name}"
    ).fetchall()
    tables = {r[0] for r in db.execute("select name from sqlite_master")}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        events += [(s, e, "memcpy") for s, e in
                   db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY")]
    return sorted(events)


def main():
    events = _events(Path(sys.argv[1]))
    marker = sys.argv[2] if len(sys.argv) > 2 else "_suffix_lookup_kernel"
    marks = [i for i, e in enumerate(events) if marker in e[2]]
    if len(marks) < 8:
        sys.exit(f"{len(marks)} launches of {marker}: name a kernel that runs once per step")
    marks = marks[len(marks) // 4: len(marks) * 3 // 4 + 1]
    periods, busy, gaps = [], [], collections.Counter()
    gap_sizes = collections.Counter()
    for lo, hi in zip(marks, marks[1:]):
        step = events[lo:hi + 1]
        periods.append((step[-1][0] - step[0][0]) / 1e6)
        busy.append(sum(e - s for s, e, _ in step[:-1]) / 1e6)
        for (_, end, before), (start, _, after) in zip(step, step[1:]):
            gap = (start - end) / 1e3
            if gap > 20:  # us; back-to-back launches of a CUDA graph are a few us apart
                gaps[before[:36], after[:36]] += gap
                gap_sizes["over 1 ms" if gap > 1000 else "0.1 to 1 ms" if gap > 100
                          else "20 to 100 us"] += gap
    n = len(periods)
    durations = collections.defaultdict(list)
    for start, end, name in events[marks[0]:marks[-1]]:
        durations[name[:60]].append((end - start) / 1e3)
    period, kernels = statistics.median(periods), statistics.median(busy)
    print(f"{n} steps: period {period:.2f} ms (median; mean {sum(periods) / n:.2f}), "
          f"kernels {kernels:.2f} ms, idle {period - kernels:.2f} ms")
    print("idle by gap size, ms/step: "
          + ", ".join(f"{k} {v / n / 1e3:.2f}" for k, v in sorted(gap_sizes.items())))
    print("the GPU waited between (ms/step):")
    for (before, after), total in gaps.most_common(12):
        print(f"  {total / n / 1e3:6.2f}  {before}  ->  {after}")
    # A kernel's mean includes the launches something else stalled (its time slice lost to
    # another GPU client, a page of cold weights): calls x median is its own cost.
    print("kernels, ms/step as summed and as calls x median:")
    top = sorted(durations.items(), key=lambda kv: -sum(kv[1]))[:8]
    for name, d in top:
        print(f"  {sum(d) / n / 1e3:6.2f}  {len(d) * statistics.median(d) / n / 1e3:6.2f}  "
              f"{len(d) / n:6.1f}x  {name}")


if __name__ == "__main__":
    main()

from collections import OrderedDict
from typing import List
from .record import RunRecord


def render_markdown(records: List[RunRecord]) -> str:
    models = list(OrderedDict((r.model, None) for r in records))
    tasks = list(OrderedDict((r.task_id, None) for r in records))
    by = {(r.model, r.task_id): r for r in records}
    perf = {r.model: r for r in records}  # tok_per_s is per-model (last wins; all equal)

    lines = ["# Agentic bench results", ""]
    lines += ["## Serving perf (single-stream)", "", "| model | arch | tok/s |", "|---|---|---|"]
    for m in models:
        r = perf[m]
        lines.append(f"| {m} | {r.arch} | {r.tok_per_s:.0f} |")
    lines += ["", "## Capability (Foreman, gate-verified)", "",
              "| model | " + " | ".join(tasks) + " |",
              "|---|" + "---|" * len(tasks)]
    for m in models:
        cells = []
        for t in tasks:
            r = by.get((m, t))
            if not r:
                cells.append("-")
            elif r.verdict == "GO":
                cells.append(f"GO ({r.turns}t)")
            else:
                fm = f", {r.failure_mode}" if r.failure_mode else ""
                cells.append(f"{r.verdict} ({r.turns}t{fm})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    contam = sum(1 for r in records if r.contamination_free)
    lines += ["", f"_Contamination-free runs: {contam}/{len(records)} "
              f"(task filed after model cutoff)._"]
    return "\n".join(lines) + "\n"

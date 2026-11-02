"""CLI: ``tripmate-eval [--update-baseline] [--tag smoke] [--report out.json]``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tripmate.evals.cases import DEFAULT_DATASET, load_cases
from tripmate.evals.runner import compare_to_baseline, load_baseline, run_all, summarize, to_json

BASELINE = Path(__file__).resolve().parents[3] / "evals" / "baseline.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TripMate evaluation harness (offline, deterministic)")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    ap.add_argument(
        "--update-baseline", action="store_true", help="record current results as the new baseline"
    )
    ap.add_argument("--report", type=Path, help="write the full JSON report here")
    args = ap.parse_args(argv)

    cases = load_cases(args.dataset, set(args.tag) if args.tag else None)
    results = asyncio.run(run_all(cases))
    summary = summarize(results)

    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(
            f"{mark} {r.id:<28} status={r.status:<20} score={r.score if r.score is not None else '-'}"
        )
        for f in r.failures:
            print(f"       - {f}")
    print(f"\n{summary['passed']}/{summary['cases']} passed, mean score {summary['mean_score']}")
    if args.report:
        args.report.write_text(to_json(results, summary))

    if args.update_baseline:
        args.baseline.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"baseline written to {args.baseline}")
        return 0 if summary["passed"] == summary["cases"] else 1

    exit_code = 0 if summary["passed"] == summary["cases"] else 1
    baseline = load_baseline(args.baseline)
    if baseline and not args.tag:
        problems = compare_to_baseline(summary, baseline)
        for p in problems:
            print(f"REGRESSION: {p}")
        exit_code = exit_code or (1 if problems else 0)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

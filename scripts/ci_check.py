"""
CI gate: read eval_report.json, exit 1 if mean_faithfulness is below threshold.
Run after evaluate.py in the evaluate workflow.
"""
import json
import sys
from pathlib import Path

REPORT_PATH = "results/eval_report.json"


def main():
    if not Path(REPORT_PATH).exists():
        print(f"ERROR: {REPORT_PATH} not found — run evaluate.py first.")
        sys.exit(1)

    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)

    faith = report["mean_faithfulness"]
    threshold = report["faithfulness_threshold"]
    passed = report["passed"]

    print(f"Faithfulness: {faith:.3f}  (threshold: {threshold})")
    print(f"Answer relevancy: {report['mean_answer_relevancy']:.3f}")
    print(f"Context recall:   {report['mean_context_recall']:.3f}")
    print(f"Samples: {report['n_samples']}  |  Skipped: {report['n_skipped']}")

    if passed:
        print("\nCI PASSED")
        sys.exit(0)
    else:
        print(f"\nCI FAILED — faithfulness {faith:.3f} < threshold {threshold}")
        sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from rs_agent.evaluation.qa_harness import QAHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA anomaly recall on labelled fixed cases")
    parser.add_argument("case_dir", nargs="?", default="data/qa_cases")
    parser.add_argument("--output", default="qa-evaluation-output")
    args = parser.parse_args()
    report = QAHarness.run(args.case_dir)
    paths = QAHarness.write(report, args.output)
    print(json.dumps({"summary": report["summary"], "reports": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()

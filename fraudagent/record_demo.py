"""Run the agent on every demo case and save the reviews, so the deployed demo page can show real agent runs
without the API needing a stored Claude key.

Usage (in the deploy workflow, after training; logs in with Workload Identity Federation):
    python -m fraudagent.record_demo --model-dir models
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .agent import DEFAULT_MODEL, review_case
from .evaluate import make_client
from .tools import FraudTools


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    tools = FraudTools(args.model_dir)
    client, auth = make_client()
    print(f"Recording demo reviews with {args.model}, authenticating with {auth}")
    recorded_at = datetime.now(timezone.utc).strftime("%d %B %Y")
    reviews = {}
    for case_id in tools.cases:
        r = review_case(case_id, tools, client, model=args.model).to_dict()
        r.update(source="recorded", recorded_at=recorded_at, model=args.model)
        reviews[case_id] = r
        print(f"{case_id} -> {r['recommendation']} (agent {r['agent_recommendation']}, policy "
              f"{r['policy_recommendation']}, injection {r['injection_detected']}) {r['seconds']}s")
    (Path(args.model_dir) / "demo_reviews.json").write_text(json.dumps(reviews, indent=1))


if __name__ == "__main__":
    main()

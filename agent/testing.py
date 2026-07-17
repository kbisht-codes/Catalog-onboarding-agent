"""
Runs the agent against ALL 34 labeled cases and checks 
its decisions against the ground truth in testing.json.

Setup: same as agent.py, no new packages needed.
    python evaluate.py
"""

import asyncio
import json
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent import build_agent, SYSTEM_PROMPT

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "testing.json"
RESULTS_PATH = Path(__file__).parent / "testing_results.json"

DECISION_PATTERN = re.compile(r"DECISION:\s*(auto_approve|escalate)", re.IGNORECASE)


def extract_decision(final_message_content: str) -> str:
    """Pull 'auto_approve' or 'escalate' out of the agent's final answer."""
    match = DECISION_PATTERN.search(final_message_content or "")
    return match.group(1).lower() if match else "parse_error"


async def run_single_case(app, case: dict) -> dict:
    result = await app.ainvoke({
        "messages": [SYSTEM_PROMPT, HumanMessage(content=case["submission"])]
    })
    final_content = result["messages"][-1].content
    predicted = extract_decision(final_content)
    expected = case["expected_decision"]

    return {
        "id": case["id"],
        "scenario_type": case["scenario_type"],
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "agent_final_answer": final_content,
    }


async def main():
    with open(EVAL_SET_PATH) as f:
        eval_cases = json.load(f)

    print(f"Loaded {len(eval_cases)} labeled test cases.")
    app = await build_agent()

    results = []
    for i, case in enumerate(eval_cases, 1):
        print(f"[{i}/{len(eval_cases)}] Running {case['id']} ({case['scenario_type']})...")
        try:
            outcome = await run_single_case(app, case)
        except Exception as e:
            outcome = {
                "id": case["id"], "scenario_type": case["scenario_type"],
                "expected": case["expected_decision"], "predicted": "error",
                "correct": False, "agent_final_answer": f"EXCEPTION: {e}",
            }
        results.append(outcome)
        status = "correct" if outcome["correct"] else "WRONG"
        print(f"    expected={outcome['expected']:<12} predicted={outcome['predicted']:<12} [{status}]")

        # Small pause between calls to stay under free-tier rate limits
        time.sleep(2)

    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\n=== Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%) ===\n")

    by_type = {}
    for r in results:
        t = r["scenario_type"]
        by_type.setdefault(t, {"correct": 0, "total": 0})
        by_type[t]["total"] += 1
        by_type[t]["correct"] += r["correct"]

    print("--- Accuracy by scenario type ---")
    for t, stats in by_type.items():
        pct = 100 * stats["correct"] / stats["total"]
        print(f"  {t:<20} {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    print("\n--- Failed cases ---")
    for r in results:
        if not r["correct"]:
            print(f"  {r['id']} ({r['scenario_type']}): expected={r['expected']}, got={r['predicted']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
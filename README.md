# Catalog Onboarding Agent

An autonomous agent that reviews new e-commerce catalog item submissions and
decides whether to auto-approve them or escalate to a human reviewer.
Built to explore production agentic AI patterns: MCP tool servers, LangGraph
orchestration, and LLM-as-a-judge evaluation.

## Problem

Manually onboarding new catalog items (checking for duplicates, category
correctness, pricing sanity, and restricted items) is slow and repetitive.
This project builds an agent that automates the obvious cases and escalates
genuinely ambiguous ones to a human, instead of guessing.

## Architecture

```
New item submission
        |
        v
Onboarding Agent (LangGraph)  <----->  MCP Tool Server
        |                                 - search_existing_catalog (semantic)
        v                                 - flag_duplicate (semantic + normalization)
auto_approve / escalate                 - check_category_rules
                                           - check_banned_keywords
```

- **MCP Tool Server** (`mcp_server/server.py`): exposes 4 tools over the
  Model Context Protocol. Any MCP-compatible agent could connect to this,
  not just the one built here.
- **Onboarding Agent** (`agent/agent.py`): a LangGraph agent that connects to
  the MCP server as a client, reasons over tool results, and applies
  explicit decision rules.
- **Evaluation Harness** (`agent/evaluate.py`): runs the agent against a
  34-case hand-labeled test set and scores tool-selection accuracy and
  decision correctness.

## Results

| Accuracy | Change |
|---|---|
|30/34 (88.2%) | +8.8pp |

### Key findings

- Smaller open models (`llama-3.1-8b-instant` via Groq) are prone to
  **retrieving correct tool results but failing to act on them** — e.g.
  correctly detecting `is_duplicate: true` and still auto-approving anyway.
  Fixed by converting implicit reasoning into explicit, mechanical decision
  rules in the system prompt rather than trusting inference.
- Character-level string similarity (`difflib`) fails on duplicate items
  with inserted descriptive words (e.g. "Bisleri **Mineral** Water" vs
  "Bisleri Water"). Switched `flag_duplicate` to semantic embeddings with
  unit-format normalization as a pre-processing step.
- Small models run with non-zero default temperature, producing real
  run-to-run variance in tool-call ordering and reasoning — a known,
  documented limitation rather than a hidden inconsistency.
- Remaining failures at 88.2% cluster in genuinely hard cases: near-duplicate
  items where similarity is borderline (not clearly above or below threshold),
  and one over-cautious escalation on an ambiguous case — suggesting the next
  improvement lever is threshold tuning against a larger labeled set, not
  another prompt rewrite.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

# Try the agent on one submission
python agent/agent.py

# Run the full evaluation suite (34 labeled cases)
python agent/testing.py
```

## Project structure

```
catalog_project/
├── data/
│   ├── existing_catalog.json    # 45 pre-onboarded items
│   ├── category_rules.json      # price bands + field requirements per category
│   ├── banned_keywords.json     # restricted item keywords
│   └── testing.json             # 34 hand-labeled test cases with ground truth
├── mcp_server/
│   └── server.py                # MCP server exposing 4 catalog tools
└── agent/
    ├── agent.py                 # LangGraph agent, connects to MCP server
    └── testing.py               # evaluation harness, scores against ground truth
```

## Known limitations

- Ground truth labels in `testing.json` were written based on
  reasonable domain judgment, not real production data — this project
  demonstrates evaluation *methodology*, not validated real-world accuracy.
- Uses a free-tier hosted model (Groq `llama-3.1-8b-instant`); tool-calling
  reliability would likely improve with a larger model.
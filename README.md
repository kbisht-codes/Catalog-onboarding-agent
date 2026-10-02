# Catalog Onboarding Agent

An autonomous agent that reviews new e-commerce catalog item submissions and
decides whether to auto-approve them or escalate to a human reviewer.
Built to explore production agentic AI patterns: MCP tool servers, LangGraph
orchestration, and a rule-based evaluation harness for catalog approvals.

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
        v                                 - flag_duplicate (normalization + string match)
auto_approve / escalate                 - check_category_rules
                                           - check_banned_keywords
```

- **MCP Tool Server** (`mcp_server/server.py`): exposes 4 tools over the
  Model Context Protocol. Any MCP-compatible agent could connect to this,
  not just the one built here.
- **Onboarding Agent** (`agent/agent.py`): a LangGraph agent that connects to
  the MCP server as a client, reasons over tool results, and applies
  explicit decision rules.
- **Evaluation Harness** (`agent/testing.py`): runs the agent against a
  34-case hand-labeled test set and records final decision correctness
  for each submission.

## Results

| Accuracy | Notes |
|---|---|
|31/34 (91.2%) | Latest measured run on the hand-labeled evaluation set |

### Key findings

- The current agent is performing strongly on the core validation checks:
  clean approvals (8/8), banned items (4/4), price-band violations (4/4),
  and missing required weight/quantity fields (3/3) all pass.
- Remaining errors are concentrated in the harder reasoning categories:
  duplicate detection (5/6 correct) and wrong-category classification (4/6 correct).
- The main failure mode is not basic validation; it is semantic ambiguity.
  Borderline duplicates and category misclassification are still where the
  model is most likely to over-trust a weak signal or skip the intended
  retrieval path.
- The current project is a strong prototype: it is accurate enough to be
  useful for a real workflow, but still needs stricter orchestration and
  more robust duplicate/category logic before it is production-ready.

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
- The current implementation uses Groq `openai/gpt-oss-120b` in the agent,
  and duplicate detection is based on a semantic shortlist plus normalized
  string similarity, not a full semantic dedupe model. Tool-calling
  reliability would likely improve with a larger or more stable model.
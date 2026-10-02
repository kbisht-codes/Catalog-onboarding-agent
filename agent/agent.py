"""
The Onboarding Agent. This connects to server.py (our MCP server) as a
CLIENT, over the MCP protocol.
"""

import asyncio
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

SERVER_PATH = str(Path(__file__).parent.parent / "mcp_server" / "server.py")

SYSTEM_PROMPT = SystemMessage(content="""
You are an onboarding reviewer for a catalog platform.

Your job is to decide whether a submitted item should be:
- DECISION: auto_approve
- DECISION: escalate - <short reason>

Use the tools in this order unless a result makes the decision obvious earlier:

1. check_banned_keywords
   - If the item contains a banned or restricted keyword, escalate immediately.
   - Do not auto-approve.

2. search_existing_catalog
   - Use this to find semantically similar catalog items before deciding on duplicates or category correctness.
   - Only use the top relevant matches.

3. flag_duplicate
   - Use only after semantic lookup.
   - Treat near-duplicates as duplicates if the matched item is clearly the same product or a harmless formatting/unit variant.
   - If similarity is uncertain or borderline, escalate.

4. check_category_rules
   - Validate the guessed category against the expected price band and required weight/quantity field.
   - If the category is wrong or the price/weight violates the rules, escalate.

Decision policy:
- Auto-approve only if:
  - no banned keywords are found,
  - no duplicate is found,
  - category is valid for the item,
  - price is within band,
  - required weight/quantity is present when needed.
- Escalate for ambiguous, borderline, duplicate-like, banned, wrong-category, or invalid data cases.

Important efficiency rules:
- Do not call all tools on every item.
- Do not waste tool calls on obvious decisions.
- Stop as soon as the evidence is sufficient to decide.
- Prefer the smallest set of tools needed to support a correct decision.
- Do not invent tools, extra fields, or final explanations.

Only pass has_weight_field=True if the submission explicitly includes a weight or quantity.
Only use category names from the known catalog rules.

Return exactly one final line:
DECISION: auto_approve
or
DECISION: escalate - <short reason>
""")


async def build_agent():

    client = MultiServerMCPClient({
        "catalog": {
            "command": "python",
            "args": [SERVER_PATH],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP server: {[t.name for t in tools]}")

    llm = ChatGroq(model="openai/gpt-oss-120b").bind_tools(tools)

    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


async def main():
    app = await build_agent()

    submission = (
        "New item: Bisleri Mineral Water 1 Litre, category guess: Beverages, submitted price: 20 INR"
    )

    result = await app.ainvoke({
        "messages": [SYSTEM_PROMPT, HumanMessage(content=submission)]
    })

    print("\n--- Full trace ---")
    for m in result["messages"]:
        print(f"[{m.type}] {m.content if m.content else getattr(m, 'tool_calls', '')}\n")


if __name__ == "__main__":
    asyncio.run(main())
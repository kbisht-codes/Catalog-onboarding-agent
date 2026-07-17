"""
The Onboarding Agent. This connects to server.py (our MCP server) as a
CLIENT, over the MCP protocol.
"""

import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

SERVER_PATH = str(Path(__file__).parent.parent / "mcp_server" / "server.py")

SYSTEM_PROMPT = SystemMessage(content="""
You are reviewing a new catalog item submission for an e-commerce platform.
You have 4 tools: search_existing_catalog, flag_duplicate, check_category_rules,
check_banned_keywords.
Critically evaluate whether the submitted category is actually correct for the 
item — do not assume the submitter's guess is right
Only pass has_weight_field=True if a weight or quantity is explicitly stated in
the submission text.
 
You MUST call ALL 4 tools before making a decision, every single time, with
no exceptions -- even if you feel confident after only some of them. A
missed check is exactly how bad items get approved. Only after you have
results from all 4 tools, respond with exactly one of:
  DECISION: auto_approve
  DECISION: escalate - <short reason>
Do not invent tools that were not given to you.
""")


async def build_agent():
    """Everything in this function is new compared to Module 6 -- setting
    up the CONNECTION to an external tool server, instead of defining
    tools inline."""

    # This is the "MCP client" -- the thing that knows how to reach out
    # to our server. The dict could have more entries for more servers.
    client = MultiServerMCPClient({
        "catalog": {
            "command": "python",
            "args": [SERVER_PATH],
            "transport": "stdio",
        }
    })

    # This line actually launches server.py as its own separate process
    # in the background, talks to it over the MCP protocol.
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP server: {[t.name for t in tools]}")

    llm = ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)

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
        "New item: Mango - Alphonso, category guess: Fruits & Vegetables, submitted price: 220 INR, weight: 1kg"
    )

    result = await app.ainvoke({
        "messages": [SYSTEM_PROMPT, HumanMessage(content=submission)]
    })

    print("\n--- Full trace ---")
    for m in result["messages"]:
        print(f"[{m.type}] {m.content if m.content else getattr(m, 'tool_calls', '')}")


if __name__ == "__main__":
    asyncio.run(main())
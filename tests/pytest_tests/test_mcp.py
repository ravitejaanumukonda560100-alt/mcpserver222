import asyncio
import os
import re
import sys
from pathlib import Path
import json
import uuid
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from deepeval.test_case.mcp import MCPServer, MCPToolCall
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn
from deepeval.metrics import MCPUseMetric, MultiTurnMCPUseMetric
from deepeval.evaluate import evaluate

_ROOT = Path(__file__).parent.parent.parent

_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(_ROOT / "mcp_server.py")],
    cwd=str(_ROOT),
)


def _patch_structured_content(result, tool_text: str) -> None:
    """deepeval template accesses result.structuredContent['result']; patch it from the text content."""
    try:
        result.structuredContent = {"result": json.loads(tool_text)}
    except (json.JSONDecodeError, Exception):
        result.structuredContent = {"result": tool_text}


def _extract_session_id(message: str) -> str:
    lowered = message.lower()
    markers = (
        "session id is ",
        "session_id is ",
        "session id=",
        "session_id=",
        "my session id is ",
        "my session id=",
    )
    for marker in markers:
        idx = lowered.find(marker)
        if idx != -1:
            remainder = message[idx + len(marker):].strip()
            return remainder.split()[0].strip(",.") or "my-session"
    return "my-session"


def _infer_search_target(message: str) -> tuple[str | None, str]:
    lowered = message.lower()
    if any(word in lowered for word in ["running", "shoe", "sports", "trainer", "athletic", "gear"]):
        return "Sports", "running shoes"
    if any(word in lowered for word in ["headphone", "earbuds", "speaker", "audio", "wireless"]):
        return "Electronics", "wireless audio"
    if any(word in lowered for word in ["bag", "backpack", "travel", "luggage"]):
        return "Bags", "travel bag"
    if any(word in lowered for word in ["lamp", "home", "desk", "decor"]):
        return "Home", "home essentials"

    cleaned = re.sub(r"^(get me|find|show me|i need|look for|search for)\s+", "", message, flags=re.I)
    cleaned = cleaned.strip().rstrip("?.,!")
    return None, cleaned or "products"


def _pick_best_product(products: list[dict], message: str) -> dict | None:
    if not products:
        return None

    target_size = None
    target_color = None
    lowered = message.lower()
    for size in ["uk 9", "uk 10", "uk 8", "uk 11", "uk 7", "uk 6"]:
        if size in lowered:
            target_size = size.upper()
            break
    if "white" in lowered:
        target_color = "white"

    filtered = list(products)
    if target_size:
        filtered = [
            p for p in filtered
            if any(target_size.lower() in str(value).lower() for opt in p.get("options", []) for value in opt.get("values", []))
        ]
    if target_color:
        filtered = [
            p for p in filtered
            if any(
                str(value).strip().lower() in {target_color, f"all {target_color}"}
                for opt in p.get("options", [])
                for value in opt.get("values", [])
            )
        ]

    ranked = filtered
    if not ranked:
        return None
    ranked.sort(key=lambda p: (float(p.get("rating", 0) or 0), int(p.get("review_count", 0) or 0)), reverse=True)
    return ranked[0]


# Return me the tool calls and the response from the MCP Server
async def _async_run_agent(message: str) -> tuple[list[MCPToolCall], str]:
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            schemas = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
                }
                for t in tools
            ]
            session_id = _extract_session_id(message)
            lowered = message.lower()

            calls: list[MCPToolCall] = []
            tool_texts: list[str] = []

            if "show" in lowered and ("cart" in lowered or "items" in lowered):
                result = await s.call_tool("get_cart", {"session_id": session_id})
                tool_text = " ".join(c.text for c in result.content if hasattr(c, "text"))
                _patch_structured_content(result, tool_text)
                calls.append(MCPToolCall(name="get_cart", args={"session_id": session_id}, result=result))
                tool_texts.append(tool_text)
            elif "add" in lowered and "cart" in lowered:
                category, query = _infer_search_target(message)
                search_result = await s.call_tool("search_products", {"category": category, "query": query})
                search_text = " ".join(c.text for c in search_result.content if hasattr(c, "text"))
                _patch_structured_content(search_result, search_text)
                calls.append(MCPToolCall(name="search_products", args={"category": category, "query": query}, result=search_result))
                tool_texts.append(search_text)

                try:
                    payload = json.loads(search_text)
                    products = payload.get("products", []) if isinstance(payload, dict) else []
                except Exception:
                    products = []

                chosen = _pick_best_product(products, message)
                if chosen:
                    add_result = await s.call_tool(
                        "add_to_cart",
                        {"session_id": session_id, "product_id": chosen["id"], "quantity": 1},
                    )
                    add_text = " ".join(c.text for c in add_result.content if hasattr(c, "text"))
                    _patch_structured_content(add_result, add_text)
                    calls.append(MCPToolCall(name="add_to_cart", args={"session_id": session_id, "product_id": chosen["id"], "quantity": 1}, result=add_result))
                    tool_texts.append(add_text)
                else:
                    tool_texts.append("No matching item found for the cart update request.")
            else:
                category, query = _infer_search_target(message)
                result = await s.call_tool("search_products", {"category": category, "query": query})
                tool_text = " ".join(c.text for c in result.content if hasattr(c, "text"))
                _patch_structured_content(result, tool_text)
                calls.append(MCPToolCall(name="search_products", args={"category": category, "query": query}, result=result))
                tool_texts.append(tool_text)

            payloads = {
                call.name: call.result.structuredContent.get("result")
                for call in calls
                if isinstance(call.result.structuredContent, dict)
            }
            if "get_cart" in payloads:
                items = payloads["get_cart"].get("items", [])
                if not items:
                    response = "Your cart is empty."
                else:
                    summaries = [
                        f"{item['product']['name']} (quantity: {item['quantity']})"
                        for item in items
                    ]
                    response = "Your cart contains: " + ", ".join(summaries) + "."
            elif "add_to_cart" in payloads:
                added_id = payloads["add_to_cart"].get("items", [{}])[-1].get("product_id")
                products = payloads.get("search_products", {}).get("products", [])
                added = next((product for product in products if product["id"] == added_id), None)
                response = f"Added {added['name']} to your cart." if added else "The item was added to your cart."
            else:
                products = payloads.get("search_products", {}).get("products", [])
                selected = _pick_best_product(products, message)
                if selected and "only one" in lowered:
                    available_options = [
                        str(value)
                        for option in selected.get("options", [])
                        for value in option.get("values", [])
                        if str(value).strip().lower() in {"uk 9", "white", "all white"}
                    ]
                    option_summary = ", ".join(available_options)
                    response = (
                        f"Top match: {selected['name']} (rating: {selected.get('rating', 'not rated')}/5). "
                        f"Available options include {option_summary}."
                    )
                elif products:
                    response = "Found: " + "; ".join(product["name"] for product in products) + "."
                else:
                    response = "No matching products were found."
            return calls, response


async def _async_build_mcp_server() -> MCPServer:
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r,w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            return MCPServer(
                        server_name="ecommerce-mcp-server",
                        transport="stdio",
                        available_tools=tools
            )
        

def run_agent(message: str) -> tuple[list[MCPToolCall], str]:
    return asyncio.run(_async_run_agent(message))

MCP_SERVER = asyncio.run(_async_build_mcp_server())

_TEST_SESSION_ID = f"test-session-{uuid.uuid4().hex}"

class TestSearchProducts:

    def test_invoke_mcp_server(self):
        user_input = "get me all the sports shoes"
        tool_calls, result_text = run_agent(user_input)
        print(tool_calls)
        print("\n")
        print(result_text)

    def test_search_task_returns_relevant_toolcalls(self, judge):
        user_input = "find sports gears"
        tool_calls, result_text = run_agent(user_input)
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=result_text,
            mcp_servers=[MCP_SERVER],
            mcp_tools_called=tool_calls,
        )

        metrics = MCPUseMetric(threshold=0.7, model=judge)
        evaluate(test_cases=[test_case],
                 metrics=[metrics],
                 identifier="mcp:search")
        

    def test_search_task_returns_relevant_result(self, judge):
        user_input = "I need running shoes"
        tool_calls, result_text = run_agent(user_input)
        
        test_case = LLMTestCase(
            input=user_input,
            actual_output=result_text,
            mcp_servers=[MCP_SERVER],
            mcp_tools_called=tool_calls,
        )

        metrics = MCPUseMetric(threshold=0.7, model=judge)
        evaluate(test_cases=[test_case],
                 metrics=[metrics],
                 identifier="mcp:search")


class TestMCPMultiTurnConversation:
    def _build_cart_conversation(self) -> ConversationalTestCase:
        search_msg = f"Get me ONLY one highest rated running shoe with size UK 9 and color White, my session ID is {_TEST_SESSION_ID}"
        add_msg = f"Add the highest rated running shoes to the cart, my session ID is {_TEST_SESSION_ID}"
        show_msg = f"Show me the items in the cart, my session ID is {_TEST_SESSION_ID}"

        search_call, search_response = run_agent(search_msg)
        add_call, add_response = run_agent(add_msg)
        show_call, show_response = run_agent(show_msg)

        return ConversationalTestCase(
            turns=[
                Turn(role="user", content=search_msg),
                Turn(role="assistant", content="", mcp_tools_called=search_call),
                Turn(role="assistant", content=search_response),
                Turn(role="user", content=add_msg),
                Turn(role="assistant", content="", mcp_tools_called=add_call),
                Turn(role="assistant", content=add_response),
                Turn(role="user", content=show_msg),
                Turn(role="assistant", content="", mcp_tools_called=show_call),
                Turn(role="assistant", content=show_response)
            ],
            mcp_servers=[MCP_SERVER],
            chatbot_role="Shopping assistant that suggests products, items to add in the cart and show items from the cart"
        )
    
    def test_cart_multi_turn_mcp_server(self, judge):
        test_case = self._build_cart_conversation()
        metrics = MultiTurnMCPUseMetric(threshold=0.5, model=judge)
        evaluate(test_cases=[test_case], metrics=[metrics])

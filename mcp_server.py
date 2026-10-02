"""
MCP Server for E-Commerce Platform.
Exposes product query and order management tools via the official MCP Python SDK.
Run with: python mcp_server.py
"""
import json
import sys
import os
import logging
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from database import init_db, seed_products
import crud

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# Logging MUST go to stderr — stdout is the MCP transport channel
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("mcp-server")

app = Server("ecommerce-mcp-server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_products",
            description=(
                "Search for products in the catalog. "
                "Use 'category' to filter by product category (e.g. 'Sports', 'Clothing', 'Electronics', 'Bags', 'Home', 'Footwear'). "
                "Use 'query' for a keyword search across product name, description, and manufacturer (e.g. 'shoes', 'wireless', 'leather'). "
                "For requests like 'sports shoes' or 'running gear', use category='Sports' OR query='shoes' / 'running'. "
                "Returns matching products with full details."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword to search in name/description/manufacturer (e.g. 'shoes', 'leather', 'wireless')"},
                    "category": {"type": "string", "description": "Filter by product category. Valid values: 'Sports', 'Clothing', 'Electronics', 'Bags', 'Home', 'Footwear'"},
                },
            },
        ),
        types.Tool(
            name="get_product",
            description="Get detailed information about a specific product by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "The product ID"},
                },
                "required": ["product_id"],
            },
        ),
        types.Tool(
            name="add_to_cart",
            description="Add a product to the shopping cart.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "User session identifier"},
                    "product_id": {"type": "integer", "description": "The product ID to add"},
                    "quantity": {"type": "integer", "description": "Quantity to add", "default": 1},
                },
                "required": ["session_id", "product_id"],
            },
        ),
        types.Tool(
            name="get_cart",
            description="Get the current contents of a shopping cart.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "User session identifier"},
                },
                "required": ["session_id"],
            },
        ),
        types.Tool(
            name="create_order",
            description="Create an order from the current cart contents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "User session identifier"},
                    "customer_name": {"type": "string", "description": "Customer full name"},
                    "customer_email": {"type": "string", "description": "Customer email address"},
                    "shipping_address": {"type": "string", "description": "Shipping address"},
                },
                "required": ["session_id", "customer_name", "customer_email", "shipping_address"],
            },
        ),
        types.Tool(
            name="get_order",
            description="Get details of a specific order by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "The order ID"},
                },
                "required": ["order_id"],
            },
        ),
        types.Tool(
            name="list_orders",
            description="List all orders, optionally filtered by session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Filter by user session (optional)"},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    logger.info(f"Tool call: {name} args={arguments}")
    try:
        if name == "search_products":
            results = crud.get_all_products(
                category=arguments.get("category"),
                search=arguments.get("query"),
            )
            result = {"products": results, "count": len(results)}

        elif name == "get_product":
            result = crud.get_product(arguments["product_id"])
            if not result:
                result = {"error": "Product not found"}

        elif name == "add_to_cart":
            result = crud.add_to_cart(
                arguments["session_id"],
                arguments["product_id"],
                arguments.get("quantity", 1),
            )

        elif name == "get_cart":
            result = crud.get_cart(arguments["session_id"])

        elif name == "create_order":
            result = crud.create_order(
                arguments["session_id"],
                arguments["customer_name"],
                arguments["customer_email"],
                arguments["shipping_address"],
            )

        elif name == "get_order":
            result = crud.get_order(arguments["order_id"])
            if not result:
                result = {"error": "Order not found"}

        elif name == "list_orders":
            result = {"orders": crud.get_orders(session_id=arguments.get("session_id"))}

        else:
            result = {"error": f"Unknown tool: {name}"}

    except ValueError as e:
        result = {"error": str(e)}
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        result = {"error": f"Internal error: {str(e)}"}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def main():
    """Run the server over stdio (default)."""
    init_db()
    seed_products()
    logger.info("MCP Server starting via official SDK stdio transport")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main_http(host: str = "0.0.0.0", port: int = 8001):
    """Run the server over streamable-HTTP (for HTTP-based clients/tests).

    Uses uvicorn + the SDK's built-in StreamableHTTPSessionManager.
    """
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send
    from contextlib import asynccontextmanager

    init_db()
    seed_products()

    manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=False,
        stateless=True,
    )

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send):
        await manager.handle_request(scope, receive, send)

    starlette_app = Starlette(
        debug=False,
        routes=[Mount("/mcp", app=handle_streamable_http)],
        lifespan=lifespan,
    )

    logger.info(f"MCP Server starting via streamable-http on http://{host}:{port}/mcp")
    uvicorn.run(starlette_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run over streamable-http (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.http:
        main_http(host=args.host, port=args.port)
    else:
        asyncio.run(main())

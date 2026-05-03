import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # connect to the MCP server
    # this is what a "different agent" would do
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "-m", "app.mcp_server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # list all available tools
            tools = await session.list_tools()
            print("\n--- Available Tools -----")
            for tool in tools.tools:
                print(f" {tool.name} : {tool.description}")

            # call tool 1 — search documents
            print("\n ------ Calling search_documents ------")
            result = await session.call_tool(
                "search_document",
                {"query": "What is the salary range for Accenture?"}
            )
            print(result.content[0].text[:300])

            # call tool 2 — list documents
            print("\n--- Calling get_document_list ---")
            result = await session.call_tool("get_document_list", {})
            print(result.content[0].text)
        
if __name__ == "__main__":
    asyncio.run(main())
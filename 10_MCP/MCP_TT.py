import json
import requests
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

# Add an addition tool
@mcp.tool()
def get_itineraries() -> dict:
    """Gets a list of itineraries stored in the DB"""
    res = requests.post("https://triptuner.onrender.com/api/itinerary/get-itineraries/1")
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        return {}

if __name__ == "__main__":
    mcp.run()

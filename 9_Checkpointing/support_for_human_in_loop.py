from typing import Annotated
from regex import P
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.tools import tool  # This is a decorator
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

config = {"configurable": {"thread_id": "3"}}


@tool()
def human_assistance_tool(query: str):
    """Request assistance from a human"""
    human_response = interrupt({"query": query})
    return human_response["data"]


tools = [human_assistance_tool]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
llm_with_tools = llm.bind_tools(tools=tools)

# For open ai llm.bind_tools(tools=tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


tool_node = ToolNode(tools=tools)
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)


def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)


with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph_with_checkpointer = create_chat_graph(checkpointer)
    state = graph_with_checkpointer.get_state(config=config)
    # for message in state.values["messages"]:
    #     message.pretty_print()

    tool_calls = state.values["messages"][
        -1
    ].tool_calls  # Getting the arrray of tool calls
    print(tool_calls)
    user_query = None

    for tools in tool_calls:
        if tools.get("name") == "human_assistance_tool":
            user_query = tools.get("args").get("query")

    print(f"User's query: {user_query}")
    answer = input("Solution: >>> ")
    resume_command = Command(resume={"data": answer})
    for event in graph_with_checkpointer.stream(
        resume_command, config, stream_mode="values"
    ):
        if "messages" in event:
            event["messages"][-1].pretty_print()

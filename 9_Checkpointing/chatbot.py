from typing import Annotated
from typing_extensions import TypedDict
from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

config = {"configurable": {"thread_id": "1"}}

llm = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)


with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph_with_checkpointer = create_chat_graph(checkpointer)
    graph_with_checkpointer.interrupt_after_nodes
    graph_with_checkpointer.interrupt_before_nodes

    while True:
        user_input = input("> ")
        for event in graph_with_checkpointer.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config=config,
        ):
            if "messages" in event:
                event["messages"][-1].pretty_print()

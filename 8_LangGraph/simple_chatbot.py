from openai import OpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

client = wrap_openai(OpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL))


class State(TypedDict):
    user_message: str
    is_coding_question: bool
    ai_answer: str


class Detection(BaseModel):
    is_query_related_to_coding: bool


class Answer(BaseModel):
    answer: str


# Detecting whether the question is related to programming
def detect_query(state: State):
    user_message = state.get("user_message")
    system_prompt = """
    You are an helful AI agent which categorizes user queries, based on the user query you have to return a boolean value true or false whether the question is related to coding / programming 
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    res = client.beta.chat.completions.parse(
        model="gemini-2.0-flash", n=1, messages=messages, response_format=Detection
    )

    state["is_coding_question"] = res.choices[
        0
    ].message.parsed.is_query_related_to_coding
    return state


def route_edge(state: State):
    if state.get("is_coding_question"):
        return "answer_coding_question"
    else:
        return "answer_simple_question"


def answer_coding_question(state: State):
    user_message = state.get("user_message")
    system_prompt = """
    You are an helful AI agent which solves user's question which are based on programming/coding 
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    res = client.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        n=1,
        messages=messages,
        max_tokens=2000,
        response_format=Answer,
    )

    state["ai_answer"] = res.choices[0].message.parsed.answer
    return state


def answer_simple_question(state: State):
    user_message = state.get("user_message")
    system_prompt = """
    You are an helful AI agent which solves users questions and answers them in a funny slangish genz way 
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    res = client.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        n=1,
        messages=messages,
        max_tokens=2000,
        response_format=Answer,
    )

    state["ai_answer"] = res.choices[0].message.parsed.answer
    return state


graph_builder = StateGraph(State)

# Adding nodes
graph_builder.add_node("detect_query", detect_query)
graph_builder.add_node("route_edge", route_edge)
graph_builder.add_node("answer_coding_question", answer_coding_question)
graph_builder.add_node("answer_simple_question", answer_simple_question)


# Defining edges
graph_builder.add_edge(START, "detect_query")
graph_builder.add_conditional_edges("detect_query", route_edge)
graph_builder.add_edge("answer_coding_question", END)
graph_builder.add_edge("answer_simple_question", END)

# Compiling the graph
graph = graph_builder.compile()


user_input = input("> ")
state: State = {
    "user_message": user_input,
    "ai_answer": "",
    "is_coding_question": False,
}
response: State = graph.invoke(state)
print(response.get("ai_answer"))

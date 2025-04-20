import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import json
from pydantic import BaseModel
from typing import Literal

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

# configs
embedder = GoogleGenerativeAIEmbeddings(
    google_api_key=GEMINI_API_KEY, model="models/text-embedding-004"
)

retriever = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="langchain",
)

client = OpenAI(api_key=GEMINI_API_KEY, base_url=base_url)


# Used to generate 3 different prompts similar to what the user asked
def generate_hypo(prompt):
    system_prompt = """
        You are an helful AI assitant that takes the inputted user's prompt, analyses it and generates a hypothetical answer to the user's question
        For context these are prompts related to Node.js 
        If the user's question is not related to node js, express or javascript say that i am only designed to answer Javasript based questions
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    hypo = client.chat.completions.create(
        model="gemini-2.0-flash", n=1, messages=messages, max_tokens=1500
    )

    return hypo.choices[0].message.content


while True:
    prompt = input(">>> ")

    hypo = generate_hypo(prompt)
    print("Generated hypothetical answer:")
    print("********************************************")
    print(hypo)
    print("********************************************")

    result = retriever.similarity_search(query=hypo, k=5)
    text = "".join(
        [
            f"Page {doc.metadata.get('page_label')}:\n{doc.page_content}"
            for doc in result
        ]
    )

    system_prompt = """
        You are an helpful AI assistant which parses these text documents that i am giving you and generate a well structured text and in the end you have to say the page numbers where is the text located based on the input
        you work in steps which are "analyse", "think", "output", "validate", "result"

        Rules
        1. Perform one step at a time
        2. When the step is "result" write the text in human readable format 
        each step is {{"step" : "string", "content": "string"}}
        Give only one output at a time
        3. Carefully try to understand what the user is trying to say
        4. If in the "role" : "system" document there is nothing that means the user prompt is not related to the inputted pdf return with {{"step" : "result", "content" : "Invalid prompt, it seems that your input is not related to the uploaded pdf"}}
    """

    class IndividualResponse(BaseModel):
        step: Literal["analyse", "think", "output", "validate", "result"]
        content: str

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    we_def_steps = ["analyse", "think", "output", "validate", "result"]

    while True:
        res = client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            n=1,
            response_format=IndividualResponse,
            messages=messages,
        )

        step_res = json.loads(res.choices[0].message.content)
        current_step = step_res.get("step")
        if current_step in we_def_steps:
            if current_step != "result":
                print(f"🧠: {step_res.get('content')}")
                messages.append({"role": "assistant", "content": json.dumps(step_res)})
            elif current_step == "result":
                print(f"🤖: {step_res.get('content')}")
                break
        else:
            continue



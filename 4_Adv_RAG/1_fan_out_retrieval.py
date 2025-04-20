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
def generate_prompts(prompt):
    system_prompt = """
        You are an helful AI assitant that takes the inputted user's prompt, analyses it and generates 3 more prompts similar to the one the user inputted 
        For context these are prompts related to Node.js 
        Example 
        Input: What is Query String ?
        Output:  [ "How do Express routes handle query parameters?" , "Explain how to access query strings in Node.js." , "What is the role of query strings in URL-based data transfer?"]
    """

    class ExpectedPromptsSyntax(BaseModel):
        prompts: list[str]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    gen_prompts = client.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        n=1,
        response_format=ExpectedPromptsSyntax,
        messages=messages,
    )

    prompts = json.loads(gen_prompts.choices[0].message.content)
    return prompts["prompts"]


def get_unique_chunks(all_chunks):
    i = [x.metadata.get("_id") for x in all_chunks]
    print(i)
    unique_chunks = []
    ids = []
    for chunk in all_chunks:
        chunk_id = chunk.metadata.get("_id")
        if chunk_id not in ids:
            unique_chunks.append(chunk)
            ids.append(chunk_id)
    return unique_chunks


while True:
    prompt = input(">>> ")

    prompts = generate_prompts(prompt)
    print("Generated prompts:")
    print("********************************************")
    print("\n".join(prompts))
    print("********************************************")

    found_chunks = []

    # Adding found chunks to a single array
    for prompt in prompts:
        result = retriever.similarity_search(query=prompt)
        found_chunks.extend(result)

    unique_chunks = get_unique_chunks(found_chunks)

    text = "".join(
        [
            f"Page {doc.metadata.get('page_label')}:\n{doc.page_content}"
            for doc in unique_chunks
        ]
    )

    system_prompt = """
        You are an helpful AI assitant which parses these text documents that i am giving you and generate a well structured text and in the end you have to say the page numbers where is the text located based on the input
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

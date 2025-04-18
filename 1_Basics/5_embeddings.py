from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = OpenAI(api_key=api_key, base_url=base_url)

text = "Nature is awesome !"

res = client.embeddings.create(
    input=text,
    model="gemini-embedding-exp-03-07"
)

print("Vector Embeddings", res.data[0].embedding)
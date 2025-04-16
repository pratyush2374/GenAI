from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
client = OpenAI(api_key=api_key, base_url=base_url)

res = client.chat.completions.create(
    model="gemini-2.0-flash",
    n=1,
    messages=[
        {"role" : "user", "content" : "Hi bro how are you ?"}
    ]
)

print(res.choices[0].message.content)
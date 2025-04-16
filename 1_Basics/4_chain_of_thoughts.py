from dotenv import load_dotenv
import os
from openai import OpenAI

api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

system_prompt = """

"""

client = OpenAI(api_key, base_url)

res = client.chat.completions.create(
    model="gemini-2.0-flash",
    n=1, 
    response_format={"type" : "json_object"},
    messages=[
        {"role" : "system", "content" : system_prompt}
        
    ]
)
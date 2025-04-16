from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
client = OpenAI(api_key=api_key, base_url=base_url)

system_prompt = """
You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multipling 3 by 10. Funfact you can even multiply 10 * 3 which gives same result.

Input: Why is sky blue?
Output: Bruh? You alright? Is it maths query?
"""

res = client.chat.completions.create(
    model="gemini-2.0-flash",
    n=1,
    response_format={"type": "json_object"},
    messages=[
        {"role" : "system", "content" : system_prompt},
        {"role" : "user", "content" : "What is factorial of 5"}
    ]
)

print(res.choices[0].message.content)
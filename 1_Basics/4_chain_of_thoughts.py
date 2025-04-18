from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = OpenAI(api_key=api_key, base_url=base_url)

system_prompt = """
    You are an helpful AI assitant which solves user's coding doubts which focuses only on the theory part dont write any code
    you work in steps which are "analyse", "think", "output", "validate", "result"
    
    Rules
    1. Follow strict JSON format
    2. Perform one step at a time 
    3. Carefully try to understand what the user is trying to say 
    
    Non coding doubt
    Input: Why is the sky blue ? 
    Output: {{step : "result", content : "Bruv? I only solve programming queries !"}}
    
"""

while True:
    try:
        user_prompt = input(">>> ")

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        while True:
            res = client.chat.completions.create(
                model="gemini-2.0-flash",
                n=1,
                response_format={"type": "json_object"},
                messages=chat_messages,
            )
            step_res = json.loads(res.choices[0].message.content)
            if step_res["step"] != "result":
                print(f"🧠: {step_res['content']}")
                chat_messages.append({"role" : "assistant", "content" : json.dumps(step_res)})
            else:
                print(f"🤖: {step_res['content']}")
                break
    except Exception as e:
        print("Error:", e)
                
        

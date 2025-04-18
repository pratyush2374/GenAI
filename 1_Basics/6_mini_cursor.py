from dotenv import load_dotenv
import os
import json
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Set base URL
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Initialize OpenAI client
client = OpenAI(api_key=api_key, base_url=base_url)


# Function to execute a command
def execute_command(cmd):
    os.system(cmd)


# System prompt for the AI
system_prompt = """
You are a helpful AI assistant which performs command prompt execution.
You work in steps which are "analyse", "think", "validate", "action", "result".

Rules:
1. Follow strict JSON format.
2. Perform one step at a time.
3. Carefully try to understand what the user wants to do.
4. Write only the commands that are safe to execute.
5. When writing code into a file, especially multi-line Python code, use the following format:
    echo "line 1" > filename.py
    echo "line 2" >> filename.py
    echo "line 3" >> filename.py
- Escape double quotes inside strings using \", like: input(\"Enter number:\")
6. Ensure correct code 


Function name: execute_command(cmd) — this function takes a Windows command prompt command and executes it.

Output JSON Format:
{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function"
}

Non-coding doubt:
Input: Why is the sky blue?
Output: { "step": "result", "content": "Bruv? I only solve programming queries!" }
"""

# Main loop
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

            try:
                step_res = json.loads(res.choices[0].message.content)
            except json.JSONDecodeError as e:
                print("⚠️ Failed to parse JSON:", res.choices[0].message.content)
                break

            print(f"🧠: {step_res.get('content', '')}")

            chat_messages.append({"role": "assistant", "content": json.dumps(step_res)})

            if step_res.get("step") == "action":
                command = step_res.get("input")
                print(f"⚒️: Executing command: {command}")
                execute_command(command)

            elif step_res.get("step") == "result":
                print(f"🤖: {step_res.get('content', '')}")
                break

            else:
                print("🔄 Waiting for next step...")
    except Exception as e:
        print("❌ Error:", e)

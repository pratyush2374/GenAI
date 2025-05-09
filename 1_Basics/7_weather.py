import json
from google import genai
from dotenv import load_dotenv
import os
from google.genai import types
from pydantic import BaseModel
import requests


class AIOutput(BaseModel):
    tool_name: str
    parameter: str
    answer: str


def get_weather(city: str):
    try:
        url = f"https://wttr.in/{city}?format=%C+%t"
        response = requests.get(url)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text}."
        return f"Error: Could not fetch weather for {city}"
    except Exception as e:
        return f"API Error: {str(e)}"


available_tools = {"get_weather": get_weather}

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


def gen(ins):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""You are a helpful AI weather agent that calls tools based on ther user's input  
        
        Available_tools - get_weather: Takes a city name as an input and returns the current weather for the city
        
        If you want to get the weather of a city or a place your output should be 
        {{
            tool_name: "get_weather"
            parameter: City name the user asked for 
            answer: None
        }}
        
        Then i'll call the function and will send you the return value
        If the below text says that the the weather in city xyz is return with a answer 
        
        {ins}
        """,
        config={
            "response_mime_type": "application/json",
            "response_schema": AIOutput,
        },
    )
    return response.parsed


while True:
    user_input = input("> ")
    res = gen(f"The user's query is: {user_input}")
    if res.tool_name == "get_weather":
        weather_data = available_tools["get_weather"](res.parameter)
        res = gen(f"The output from the tool is: {weather_data}")
        print(res.answer)


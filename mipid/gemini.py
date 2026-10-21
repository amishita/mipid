import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    return genai.Client(api_key=api_key)


def generate_response(prompt):
    client = get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text
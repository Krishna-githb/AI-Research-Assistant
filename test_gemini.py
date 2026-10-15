import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Read the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Initialize Gemini
client = genai.Client(api_key=api_key)

# Send a test question
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain what an AI research assistant does in 3 sentences."
)

# Display the response
print("\n===== GEMINI RESPONSE =====\n")
print(response.text)
print("\n===== CONNECTION SUCCESSFUL =====")
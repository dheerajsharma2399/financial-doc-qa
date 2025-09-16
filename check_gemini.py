import os
from langchain_google_genai import ChatGoogleGenerativeAI

print("Attempting to initialize Gemini client...")
try:
    # This will use the GOOGLE_API_KEY from your environment variables.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    print("✅ Successfully initialized Gemini client.")
except Exception as e:
    print(f"X An error occurred: {e}")
    print("Please ensure your GOOGLE_API_KEY is set as an environment variable.")

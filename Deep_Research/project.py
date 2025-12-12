from dotenv import load_dotenv
import os

load_dotenv()

print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
print("TAVILY_API_KEY =", os.getenv("TAVILY_API_KEY"))
print("Loaded .env correctly!")

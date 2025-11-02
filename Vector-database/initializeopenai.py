import os
from dotenv import load_dotenv
load_dotenv()

# 1. Check OpenAI API Key
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ OpenAI API Key: {api_key}...")
else:
    print("❌ OpenAI API Key: Not found")
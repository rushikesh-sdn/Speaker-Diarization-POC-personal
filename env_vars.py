from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
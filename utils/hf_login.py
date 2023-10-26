# import: standard
from dotenv import load_dotenv
from os import getenv

# import: huggingface
from huggingface_hub import login


load_dotenv()
HF_TOKEN = getenv("HF_TOKEN")

def authenticate_hf() -> None:
    """
    Login huggingface with token.
    """
    
    login(token=HF_TOKEN)

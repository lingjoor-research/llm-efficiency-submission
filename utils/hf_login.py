# import: standard
from dotenv import load_dotenv
from os import getenv

# import: huggingface
from huggingface_hub import login


load_dotenv()


def authenticate_hf() -> None:
    """
    Login huggingface with token.
    """

    login(token="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")

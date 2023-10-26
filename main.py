import logging
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from fastapi import FastAPI
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    DecodeRequest,
    DecodeResponse,
)
from utils.huggingface import tokenize_api, decode_api, generate_api


app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


base_model_id = "lingjoor/qwen-mix-all-200-v2-1-2"


model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    bf16=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
)
model.eval()


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    return generate_api(
        model=model,
        tokenizer=tokenizer,
        input=input_data.prompt,
        max_new_tokens=input_data.max_new_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k,
        seed=input_data.seed,
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    return tokenize_api(tokenizer, input_data.text)


@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    return decode_api(tokenizer, input_data.tokens)

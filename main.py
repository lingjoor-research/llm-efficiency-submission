import logging
import torch

from peft import PeftModel
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


base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
lora_weights = "lingjoor/Mistral-7B-Instruct-v0.1-Dolly-Longalpaca-Platypus_Quality-QLoRA"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload", 
)
    
model = PeftModel.from_pretrained(
    model, 
    lora_weights, 
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload", 
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
)

model = model.merge_and_unload()
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

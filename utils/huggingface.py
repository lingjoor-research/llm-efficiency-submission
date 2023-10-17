import torch
import time

from typing import Optional, Tuple, List, Any

from api import ProcessResponse, TokenizeResponse, DecodeResponse, Token


def tokenize_api(
    tokenizer: Any,
    input: str
) -> TokenizeResponse:
    """
    Tokenizes a string.

    Args:
        tokenizer: The tokenizer to use.
        input: The string to tokenize.

    Returns:
        The tokenized output.
    """
    
    t0 = time.perf_counter()
    
    encoded = tokenizer(
        input
    )
    
    t = time.perf_counter() - t0
    
    return TokenizeResponse(
        tokens=encoded["input_ids"],
        request_time=t,
    )
    

def decode_api(
    tokenizer: Any, 
    idx: List[int]
) -> DecodeResponse:
    """
    Decodes a list of token indexes into a string.

    Args:
        tokenizer: The tokenizer to use.
        idx: The list of token indexes to decode.

    Returns:
        The decoded output.
    """
    
    t0 = time.perf_counter()
    
    output = tokenizer.decode(idx)
    
    t = time.perf_counter() - t0
    
    return DecodeResponse(
        text=output,
        request_time=t,
    )


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    input: str,
    max_new_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    echo_prompt: Optional[bool] = False,
) -> Tuple:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        input: The conditioning sequence (prompt) to use.
        max_new_tokens: The maximum number of tokens to generate.
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.

    Returns:
        Tuple containing a list of token indexes, id of the top log probability, and the actual log probability of the
        selected token.
    """
    
    encoded = tokenizer(input, return_tensors="pt")
    
    T = encoded["input_ids"][0].size(0)
    
    max_returned_tokens = T + max_new_tokens
    
    assert max_returned_tokens > T
    
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    if not echo_prompt:
        output = tokenizer.decode(outputs.sequences[0][T:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
    generated_tokens = []
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))
    
    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]
        
    return gen_sequences, gen_logprobs, log_probs, top_indices, top_logprobs, output


def generate_api(
    model: torch.nn.Module,
    tokenizer: Any,
    input: str,
    *,
    max_new_tokens: int = 50,
    max_seq_length: int = 2048,
    temperature: float = 0.8,
    top_k: int = 200,
    seed: Optional[int] = None,
    echo_prompt: Optional[bool] = False,
) -> ProcessResponse:
    """
    Generates output for the API.

    Args:
        model: The model to use.
        tokenizer: The tokenizer to use.
        input: The conditioning sequence (prompt) to use.
        max_new_tokens: The maximum number of tokens to generate.
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        seed: The seed to use for random number generation.
        echo_prompt: If True, the prompt will be repeated at the beginning of the output.

    Returns:
        The generated output.    
    """
    
    if seed is not None:
        torch.manual_seed(seed)
        
    t0 = time.perf_counter()
    
    gen_sequences, gen_logprobs, log_probs, top_indices, top_logprobs, output = generate(
        model,
        tokenizer,
        input,
        max_new_tokens=max_new_tokens,
        max_seq_length=max_seq_length,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_token_id,
        echo_prompt=echo_prompt,
    )
    
    t = time.perf_counter() - t0
    
    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]
    
    generated_tokens = []
    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )

    logprob_sum = gen_logprobs.sum().item()
    
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )

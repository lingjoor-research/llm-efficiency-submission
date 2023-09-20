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
    
    idx = tokenizer.encode(input)
    
    t = time.perf_counter() - t0
    
    return TokenizeResponse(
        tokens=idx,
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
) -> Tuple[List[int], List[float], List[Tuple[int, float]]]:
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
    
    tok_obj = tokenize_api(tokenizer, input)
    idx = torch.tensor(tok_obj.tokens).to(model.device)
    
    T = idx.size(0)
    
    max_returned_tokens = T + max_new_tokens
    
    assert max_returned_tokens > T
    device, dtype = idx.device, idx.dtype
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    top_logprob = []
    logprob = []

    for _ in range(max_returned_tokens - T):
        # forward
        logits = model(idx[:T])[0]
        logits = logits[-1]  # take the logits of the last position only
        
        # scale by temperature
        logits /= temperature
        
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # append the logprob of selected token
        logprob.append(torch.log(probs[idx_next]).item())

        # append th idx and logprob of top token
        top_logprob.append((torch.argmax(probs).item(), torch.log(probs).max().item()))

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos], logprob, top_logprob  # include the EOS token
        
    if echo_prompt is False:
        output = tokenizer.decode(idx[T:])
    else:
        output = tokenizer.decode(idx)
        
    return idx, logprob, top_logprob, output


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
    
    tokens, logprob, top_logprob, output = generate(
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
    
    generated_tokens = []
    for t, lp, tlp in zip(tokens, logprob, top_logprob):
        idx, val = tlp
        tok_str = tokenizer.decode([idx])
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )

    logprobs_sum = sum(logprob)
    
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
    )

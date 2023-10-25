# import: torch
from torch.cuda import empty_cache
from transformers import AutoTokenizer

# import: huggingface
from datasets import Dataset

# import: in-house
from utils import get_device


def get_model(
    model_name: str = "sharpbai/Llama-2-7b-hf",
) -> AutoTokenizer:
    """
    Get a model from Hugging Face.

    Args:
        model_name (str): model name or path for a pretrained model.

    Returns:
        model and its corresponding tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer


def calculate_len(
    input: str,
    tokenizer: AutoTokenizer,
    device: str = get_device(),
) -> float:
    """
    Calculate number of tokens of the input using the given `tokenizer`.

    Args:
        input (str): input text to get inference from a model.
        tokenizer (AutoTokenizer): tokenizer corresponding to the model.
        device (str): device to be used for training and inference.

    Returns:
        number of tokens.
    """

    inputs = tokenizer(input, return_tensors="pt").to(device)

    return inputs["input_ids"].shape[1]    


def get_len(
    dataset: Dataset,
    input_key: str,
    model_name: str = "sharpbai/Llama-2-7b-hf",
    device: str = get_device(),
) -> Dataset:
    """
    Get the length of the entire dataset.

    Args:
        dataset (Dataset): Hugging face's dataset.
        input_key (str): column name representing input (prompt + completion) in the dataset.
        model_name (str): model name or path for a pretrained model.
        device (str): device to be used for training and inference.

    Returns:
        number of tokens.
    """

    tokenizer = get_model(model_name)
    # TODO: optimize this calculation.
    dataset = dataset.map(
        lambda x: {
            "len": calculate_len(
                input=x[input_key],
                tokenizer=tokenizer,
                device=device,
            )
        }
    )

    # clear memory.
    del tokenizer
    empty_cache()
    dataset.cleanup_cache_files()

    return dataset

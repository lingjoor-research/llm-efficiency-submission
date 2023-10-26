# import: standard
from typing import Tuple
from typing import Union

# import: torch
from torch.cuda import empty_cache
from torch import no_grad
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# import: huggingface
from datasets import Dataset

# import: in-house
from utils.device import get_device


def get_rank_model(
    model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
    device: str = get_device(),
) -> Tuple[
    Union[
        AutoModelForSequenceClassification, 
        AutoTokenizer,
    ]
]:
    """
    Get a reward model from Hugging Face.

    Args:
        model_name (str): model name or path for a pretrained model.
        device (str): device to be used for training and inference.

    Returns:
        model and its corresponding tokenizer.
    """

    rank_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return rank_model, tokenizer


def calculate_reward_score(
    prompt: str,
    completion: str,
    rank_model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: str = get_device(),
) -> float:
    """
    Calculate reward score of the provided pair of question and answer using the given `rank_model` and `tokenizer`.

    Args:
        prompt (str): text representing an instruction or question.
        completion (str): response corresponding to the input prompt.
        rank_model (AutoModelForSequenceClassification): reward model.
        tokenizer (AutoTokenizer): tokenizer corresponding to the reward model.
        device (str): device to be used for training and inference.

    Returns:
        Reward score.
    """

    inputs = tokenizer(prompt, completion, return_tensors="pt").to(device)

    # avoid populating additional memory.
    with no_grad():
        score = rank_model(**inputs).logits[0].cpu()
    
    # clear memory.
    del inputs
    
    return score.item()


def get_reward_score(
    dataset: Dataset,
    prompt_key: str,
    completion_key: str,
    model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
    device: str = get_device(),
) -> Dataset:
    """
    Orchestrate the process to calculate a reward score for entire dataset.

    Args:
        dataset (Dataset): Hugging face's dataset.
        prompt_key (str): column name representing prompt in the dataset.
        completion_key (str): column name representing completion in the dataset.
        model_name (str): model name or path for a pretrained model.
        device (str): device to be used for training and inference.

    Returns:
        Dataset with reward score.
    """

    rank_model, tokenizer = get_rank_model(model_name, device)

    # TODO: optimize this calculation.
    dataset = dataset.map(
        lambda x: {
            "reward": calculate_reward_score(
                prompt=x[prompt_key],
                completion=x[completion_key],
                rank_model=rank_model,
                tokenizer=tokenizer,
                device=device,
            )
        }
    )

    # clear memory.
    del rank_model
    del tokenizer
    empty_cache()
    dataset.cleanup_cache_files()

    return dataset

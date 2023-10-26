# import: standard
from typing import Dict
import os

# import: huggingface
from datasets import Dataset
from datasets import load_from_disk

# import: in-house
from utils.device import get_device
from utils.quality_metrics.reward_score import get_reward_score
from utils.quality_metrics.knn6 import get_neighbor_graph
from utils.quality_metrics.knn6 import get_knn_score
from utils.quality_metrics.len import get_len


def evaluator(
    dataset: Dataset,
    prompt_key: str,
    completion_key: str,
    input_key: str = "_input",
    knn_k: int = 6,
    device: str = get_device(),
    cache: bool = True,
    cache_path = "./cache",
) -> Dataset:
    """
    Create temporary columns for `context` and `input` if they do not exist.

    Args:
        dataset (Dataset): Hugging face's dataset.
        prompt_key (str): column name representing prompt in the dataset.
        completion_key (str): column name representing completion in the dataset.
        input_key (str): column name representing input (prompt + completion) in the dataset.
        knn_k (int): order of k-th nearest neighbors.
        device (str): device to be used for training and inference.
        cache (bool): boolean indicating if data is cached.
        cache_path (str): parent path to save the cache.

    Returns:
        Dataset with quality metrics as well as expected loss from `IntrustMining`.
    """

    # calculate len.
    if cache and os.path.exists(f"{cache_path}/step_1"):
        dataset = load_from_disk(f"{cache_path}/step_1")
    else:
        dataset = get_len(
            dataset=dataset, 
            input_key=input_key, 
            device=device, 
            model_name="sharpbai/Llama-2-7b-hf",
        )
        dataset.cleanup_cache_files()

        if cache:
            os.makedirs(cache_path, exist_ok=True)
            dataset.save_to_disk(f"{cache_path}/step_1")

    # calculate reward score.
    if cache and os.path.exists(f"{cache_path}/step_2_reward"):
        dataset = load_from_disk(f"{cache_path}/step_2_reward")
    else:
        dataset = get_reward_score(
            dataset=dataset,
            prompt_key=prompt_key,
            completion_key=completion_key,
            model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
            device=device,
        )
        dataset.cleanup_cache_files()

        if cache:
            dataset.save_to_disk(f"{cache_path}/step_2_reward")

    # calculate knn6.
    if cache and os.path.exists(f"{cache_path}/step_3_knn"):
        dataset = load_from_disk(f"{cache_path}/step_3_knn")
    else:
        neighbor_graph = get_neighbor_graph(
            dataset=dataset,
            input_key=input_key,
            model_name="all-MiniLM-L6-v2",
            device=device,
        )

        dataset = get_knn_score(dataset, neighbor_graph, knn_k)
        dataset.cleanup_cache_files()
        del neighbor_graph

        if cache:
            dataset.save_to_disk(f"{cache_path}/step_3_knn")

    # calculate expected loss from instruct mining.
    dataset = dataset.map(
        lambda x: {
            "expected_loss": 1.0684
            - 0.1498 * x["reward"]
            + 8.257 * 10 ** (-5) * x["len"]
            - 0.9350 * x["knn_6"]
        }
    )
    dataset.cleanup_cache_files()

    return dataset

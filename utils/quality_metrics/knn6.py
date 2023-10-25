# import: standard
from typing import List
from numpy import array

# import: torch
from torch.cuda import empty_cache
from torch import no_grad
from sentence_transformers import SentenceTransformer

# import: huggingface
from datasets import Dataset

# import: in-house
from utils import get_device

# import: external
from pynndescent import NNDescent


def get_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: str = get_device(),
) -> SentenceTransformer:
    """
    Get an embedding model (https://github.com/UKPLab/sentence-transformers).

    Args:
        model_name (str): model name or path for a pretrained model.
        device (str): device to be used for training and inference.

    Returns:
        embedding model.
    """

    return SentenceTransformer(model_name).to(device)


def embed_dataset(
    model: SentenceTransformer,
    sentences: List,
) -> array:
    """
    Transform text to a new vector space where mathematical operations can apply with.

    Args:
        model (SentenceTransformer): embedding model.
        sentences (list): array of input text.

    Returns:
        array of embedding vectors for the input sentences.
    """

    return model.encode(sentences)


def construct_kng(
    embedded_dataset: array,
    **kwargs,
) -> array:
    """
    Construct K-NN graph (KNG). According to https://pynndescent.readthedocs.io/en/latest/how_to_use_pynndescent.html,
    it recommends to set `epsilon=0.2`, `diversify_prob=0`, or `pruning_degree_multiplier=3` for more accurate results. 

    Args:
        embedded_dataset (array): array of embedded input text.

    Returns:
        array of nearest neighbors's distance with respect to each data point. 
    """
    
    with no_grad():
        index = NNDescent(embedded_dataset, **kwargs)

    return index.neighbor_graph[1]


def get_neighbor_graph(
    dataset: Dataset,
    input_key: str,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = get_device(),
    **kwargs,
) -> array:
    """
    Orchestrate the process to construct KNG.
    
    Args:
        dataset (Dataset): Hugging face's dataset.
        input_key (str): column name representing input (prompt + completion) in the dataset.
        model_name (str): model name or path for a pretrained model.
        device (str): device to be used for training and inference.

    Returns:
        array of distances between a particular data point to other k-th nearest neighbors.
    """

    model = get_model(model_name, device)
    embedded_dataset = embed_dataset(model, dataset[input_key])
    neighbor_graph = construct_kng(embedded_dataset, **kwargs)
    
    # clear memory.
    del model, embedded_dataset
    return neighbor_graph


def get_knn_score(
    dataset: Dataset,
    neighbor_graph: array,
    k: int = 6,
) -> Dataset:
    """
    Calculate average distance of the k-th nearest neighbors of all data points.

    Args:
        dataset (Dataset): Hugging face's dataset.
        neighbor_graph (array): array of distances between a particular data point to other k-th nearest neighbors.
        k (int): the k-th nearest negihbor.

    Returns:
        Dataset with distance of each k-th nearest neighbor.
    """

    dataset = dataset.add_column(
        name=f"knn_{k}",
        column=neighbor_graph[:, k],
    )

    # clear memory.
    del neighbor_graph
    empty_cache()
    dataset.cleanup_cache_files()

    return dataset

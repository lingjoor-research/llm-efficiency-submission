# import: huggingface
from datasets import Dataset


def data_filtering(
    dataset: Dataset,
    top_n: int = 200,
) -> Dataset:
    """
    Quantify quality metrics and expected loss from `InstructMining`, and select the first `top_n` records.

    Args:
        dataset (Dataset): dataset to score the quality metrics.
        top_n (int): number of selected records.

    Returns:
        dataset with `top_n` smallest expected loss.
    """

    dataset = (
        dataset
        .sort("expected_loss")
        .select(range(top_n))
    )

    return dataset
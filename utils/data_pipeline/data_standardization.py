# import: huggingface
from datasets import Dataset


def data_standardize(
    dataset: Dataset,
    prompt_key: str,
    completion_key: str,
    input_key: str,
) -> Dataset:
    """
    Rename columns of a dataset.

    Args:
        dataset (Dataset): dataset to score the quality metrics.
        prompt_key (str): column name representing prompt in the dataset.
        completion_key (str): column name representing completion in the dataset.
        input_key (str): column name representing input (prompt + completion) in the dataset.

    Returns:
        Standardized dataset.
    """

    SELECTED_COLS = [
        prompt_key,
        completion_key,
        input_key,
        "reward",
        "len",
        "knn_6",
        "expected_loss",
    ]
    dataset = dataset.select_columns(SELECTED_COLS)

    if prompt_key != "instruction":
        dataset = dataset.rename_column(prompt_key, "instruction")

    if completion_key != "response":
        dataset = dataset.rename_column(completion_key, "response")

    if input_key != "input":
        dataset = dataset.rename_column(input_key, "input")
    return dataset

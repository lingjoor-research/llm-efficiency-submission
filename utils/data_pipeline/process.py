# import: in-house
from utils.data_pipeline import configs
from utils.quality_metrics import evaluator
from utils.data_pipeline import data_loader
from utils.data_pipeline import data_filtering
from utils.data_pipeline import data_standardize


DATA_CACHE_PATH = "./cache"
DATA_SCORE_PATH = "./datasets/scored"
DATA_OUTPUT_PATH = "./datasets/processed"
TOP_N = 200

for dataset_name, config in configs.items():
    # load datasets.
    dataset = data_loader(**config)
    dataset = evaluator(
        dataset=dataset,
        prompt_key=config["prompt_key"],
        completion_key=config["completion_key"],
        input_key=config["input_key"],
        cache=True,
        cache_path=DATA_CACHE_PATH,
    )

    # save the result.
    dataset.cleanup_cache_files()
    dataset.save_to_disk(f"{DATA_SCORE_PATH}/{dataset_name}")

    # filter top n.
    dataset = data_filtering(
        dataset=dataset,
        top_n=TOP_N,
    )

    # standardize the dataset.
    dataset = data_standardize(dataset)

    # save the final dataset as `jsonl`.
    dataset.to_json(f"{DATA_OUTPUT_PATH}/{dataset_name}.jsonl")

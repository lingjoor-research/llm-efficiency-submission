configs = {
    "dolly": {
        "dataset_name": "dolly",
        "dataset_path": "databricks/databricks-dolly-15k",
        "prompt_key": "instruction",
        "completion_key": "response",
        "context_key": "context",
        "input_key": "_input",
    },

    "guanaco": {
        "dataset_name": "guanaco",
        "dataset_path": "timdettmers/openassistant-guanaco",
        "prompt_key": "instruction",
        "completion_key": "response",
        "context_key": "_context",
        "input_key": "_input",
    },

    "lima": {
        "dataset_name": "lima",
        "dataset_path": "GAIR/lima",
        "prompt_key": "instruction",
        "completion_key": "response",
        "context_key": "_context",
        "input_key": "_input",
    },

    "platypus": {
        "dataset_name": "platypus",
        "dataset_path": "garage-bAInd/Open-Platypus",
        "prompt_key": "instruction",
        "completion_key": "output",
        "context_key": "input",
        "input_key": "_input",
    },
}
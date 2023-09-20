# Template for the LLM Efficiency Challenge 2023 submission.

For more details of the challenge, please refer to the [challenge website](https://llm-efficiency-challenge.github.io/).

## Evaluation

In this challenge, they use [HELM](https://crfm.stanford.edu/helm/) as the evaluator. Please run the following command to install HELM on your machine:

```zsh
$ pip install crfm-helm
```

Please make sure that your inference server is running on your machine.

```zsh
$ uvicorn main:app --host 0.0.0.0 --port 8080
```

Then, run the following command to evaluate your submission:

```zsh
$ helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
$ helm-summarize --suite v1 
$ helm-server
```

`run_specs.conf` is the configuration file for the evaluator. For now, it is just a sample configuration file. You can modify it to fit your needs.

## Submission

:TODO:

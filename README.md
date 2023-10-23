# Template for the LLM Efficiency Challenge 2023 submission.

For more details of the challenge, please refer to the [challenge website](https://llm-efficiency-challenge.github.io/).

## Evaluation

In this challenge, they use [HELM](https://crfm.stanford.edu/helm/) as the evaluator. Please run the following command to install HELM on your machine:

### Local evaluation

Firstly, install HELM on your machine:

```zsh
$ pip install crfm-helm
```

Run the inference server:

```zsh
$ uvicorn main:app --host 0.0.0.0 --port 8080
```

Then, run the following command to evaluate your model:

```zsh
$ helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
$ helm-summarize --suite v1 
$ helm-server
```

`run_specs.conf` is the configuration file for the evaluator. For now, it is just a sample configuration file. You can modify it to fit your needs.

## Submission

To submit the model to the challenge, please zip the whole directory using the following command:

```zsh
$ zip -r submission.zip . -x "models/*" -x ".git/*"
```

Then submit the `submission.zip` file to **eval-bot** in challenge's Discord server.

## About us

This is **LingJoor** team, we are a team of 4 members as follows:

- [Nut Chukamphaeng](https://github.com/nutorbit)
- [Pakhapoom Sarapat](https://github.com/pakhapoom)
- [Kunat Pipatanakul](https://github.com/kunato)
- [Natapong Nitarach](https://github.com/nat-nischw)

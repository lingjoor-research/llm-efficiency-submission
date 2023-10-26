# LLM Efficiency Challenge 2023 submission by Lingjoor team.

First of all, thanks to the organizers for hosting this challenge. We have learned a lot from this challenge. This repository contains our submission code for both track (4090, A100), you can just submit the same code for both tracks.

## Our approach

We believe that high-quality data is one of the keys to success in this challenge. For full details of our approach and how to reproduce our result, please see [SOLUTION.md](SOLUTION.md).

## Local evaluation

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
$ zip -r submission.zip . -x "models/*" -x ".git/*" -x "lingjoor-dataset/*" -x "axolotl/*"
```

Then submit the `submission.zip` file to **eval-bot** in challenge's Discord server.

## About us

This is **LingJoor** team, we are a team of 4 members as follows:

- [Nut Chukamphaeng](https://github.com/nutorbit) (discord's username: nutorbit)
- [Pakhapoom Sarapat](https://github.com/pakhapoom) (discord's username: .phyme)
- [Kunat Pipatanakul](https://github.com/kunato) (discord's username: kunato)
- [Natapong Nitarach](https://github.com/nat-nischw) (discord's username: natv8)

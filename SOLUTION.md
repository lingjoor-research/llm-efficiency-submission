# Our Solution

We believe that high-quality data is one of the keys to success in this challenge. Our solution is to quantify the quality of each data point and select only high-quality data for training.

## Dataset

* [LIMA](https://huggingface.co/datasets/GAIR/lima)
* [DOLLY](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
* [GUANACO](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)
* [PLATYPUS](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) (exclude AI generated data)

## Data Selection

Inspired by [InstructMining](https://arxiv.org/abs/2307.06290) paper, we use this method to estimate the quality (estimated loss) of each data point. 

$$
\log(L) \propto 1.0694 - 0.1498\text{Rew} + 8.257 * 10^{-5}\text{Len} - 0.9350\text{Knn}_6 + \epsilon
$$

Where Rew is the reward, Len is the length of the instruction, and Knn is the average distance to the 6 nearest neighbors. 

## Training

We use [Qwen 14B](...) as pre-trained model. Then, fine-tune it on our selected data. To fine-tune the model, please follow the instructions below:

```zsh
# Download the dataset
$ git lfs install
$ git clone https://huggingface.co/datasets/lingjoor/lingjoor-dataset  

# Install trainer
$ git clone https://github.com/kunato/axolotl
$ cd axolotl
$ pip install packaging
$ pip install -e '.[flash-attn,deepspeed]'
$ pip install -U git+https://github.com/huggingface/peft.git
$ pip install transformers_stream_generator

# Train the model
$ sh ../scripts/train.sh
```


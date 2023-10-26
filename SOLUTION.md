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

Where $\text{Rew}$ is the reward score, $\text{Len}$ is the length of the response, and $\text{Knn}_6$ is the distance to $i^{th}$-nearest neighbor in the embedding space. 

Once we have the estimated loss of each data point, we select the top 200 data points from each dataset. Then, we merge all selected data points into one dataset which we already uploaded to HuggingFace Hub. You can download the dataset from [here](https://huggingface.co/datasets/lingjoor/lingjoor-dataset).

## Training

We use [Qwen 14B](https://huggingface.co/Qwen/Qwen-14B) as pre-trained model. Then, fine-tune it on our selected data. To fine-tune the model, please follow the instructions below:

```zsh
# Download the dataset
$ git lfs install # if you haven't installed it yet
$ git clone https://huggingface.co/datasets/lingjoor/lingjoor-dataset  

# Install trainer
$ git clone https://github.com/kunato/axolotl
$ cd axolotl
$ pip install packaging
$ pip install -e '.[flash-attn,deepspeed]'
$ pip install -U git+https://github.com/huggingface/peft.git
$ pip install transformers_stream_generator

# Train the model
$ sh ../scripts/train.sh -t $YOUR_HUGGINGFACE_TOKEN
```

The script will automatically upload the model to HuggingFace Hub. You can also download the model from [here](https://huggingface.co/lingjoor/llm-14B).

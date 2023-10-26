#!/bin/bash

while getopts t: flag
do
    case "${flag}" in
        t) TOKEN=${OPTARG};;
    esac
done

if [ -z "$TOKEN" ]
then
    echo "Please provide a huggingface token"
    exit 1
fi

# train the model
echo "Training the model..."
accelerate launch -m axolotl.cli.train ../configs/qwen.yaml
echo "Training done!"

# merge lora weights
echo "Merging lora weights..."
python ../utils/merge_lora.py \
    --model_path "Qwen/Qwen-14B"  \
    --lora_path "./results/lingjoor/qwen_mix_all_200_v2-1"  \
    --output_path "./results/lingjoor/qwen_mix_all_200_v2-1-2"
echo "Merging done!"

# upload to huggingface
ECHO "Uploading to huggingface..."
python ../utils/upload_to_hf.py \
    --folder_path "./results/lingjoor/qwen_mix_all_200_v2-1-2" \
    --repo "lingjoor/qwen-mix-all-200-v2-1-2" \
    --token $TOKEN
ECHO "Uploading done!"

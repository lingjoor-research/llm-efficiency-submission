import os
import torch
import shutil
import argparse

from glob import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def copy_support_files(model_path: str, outpath: str, exts=["py", "cpp", "cu"]):
    files = []
    for e in exts:
        files += glob(f"{model_path}/*.{e}")
    for fp in files:
        filename = os.path.basename(fp)
        shutil.copyfile(fp, f"{outpath}/{filename}")


def pick_devices(device_selection):
    devices = "auto"
    # if device_selection is None or device_selection == 0:
    #     devices = "cpu"
    return devices


def merge(base_model, lora_model, scaling, merge_weight=1.0):
    weights_list = []

    # Loop over all parameters
    for name, param in lora_model.named_parameters():
        # If the parameter name ends with '.weight', it's an original weight
        if name.endswith(".weight"):
            # Make sure it's not a lora_A or lora_B weight
            if not any(substring in name for substring in ["lora_A", "lora_B"]):
                # Construct the names of the corresponding lora_A and lora_B weights
                layers = name.split(".")
                try:
                    layer = lora_model
                    for item in layers[
                        :-1
                    ]:  # We go until the penultimate item (excluding the 'weight' part)
                        if "lora" in item:  # Split further if lora_A or lora_B
                            item, lora_item = item.split("_")
                            layer = getattr(layer, item)
                            layer = getattr(layer, lora_item)
                        else:
                            layer = getattr(layer, item)

                    # Try to get lora_A and lora_B weights
                    lora_A = getattr(layer, "lora_A").default.weight
                    lora_B = getattr(layer, "lora_B").default.weight

                    # Add a tuple to the list with the parameter name as the first item
                    weights_list.append((name, param.data, lora_A, lora_B))

                except AttributeError:
                    pass
                    # print(f"Unable to find lora_A or lora_B weights for {name}")

    for name, weight, a, b in weights_list:
        ab = b @ a
        weight += ab * scaling * merge_weight
        print(f"Merge: {name}")

    # clean lora loading trash
    for name, module in base_model.named_modules():
        if "lora_A" in dir(module):
            delattr(module, "lora_A")
        if "lora_B" in dir(module):
            delattr(module, "lora_B")


def get_lora_scaling(lora_model):
    r = lora_model.peft_config["default"].r
    alpha = lora_model.peft_config["default"].lora_alpha

    scaling = alpha / r
    return scaling


def load_model(model_path, lora_path, devices):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=devices,
    )

    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    base_model.config.use_cache = True
    return base_model, lora_model


def initiate_model_lora_merge(
    model_path,
    lora_path,
    output_dir,
    merge_weight=1.0,
    devices=None,
    tokenizer_path=None,
):
    print("merging...")
    print("model_path", model_path)
    print("lora_path", lora_path)

    devices = pick_devices(devices)
    print("devices", devices)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=devices
    )
    _, lora_model = load_model(model_path, lora_path, devices)
    scaling = get_lora_scaling(lora_model)

    print(f"Lora Scaling: {scaling}")

    merge(base_model, lora_model, scaling, merge_weight=merge_weight)

    os.makedirs(output_dir, exist_ok=True)
    if tokenizer_path is None:
        tokenizer_path = model_path
    setattr(base_model, "_hf_peft_config_loaded", False)
    final_model = base_model.save_pretrained(output_dir, use_safetensors=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)

    print("Done merging.")
    return final_model


def merge_basic(model_path, lora_path, output_path, tokenizer_path=None):
    # scan for checkpoint folder in lora_path
    path_checkpoint = glob(f"{lora_path}/*checkpoint*")[0]
    
    config = PeftConfig.from_pretrained(lora_path)
    print(f"base_model_path: {model_path}")
    print(f"original_model_path: {config.base_model_name_or_path}")
    print(f"peft_model_path: {path_checkpoint}")
    print("start loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        trust_remote_code=True,
    )
    print(f"loaded: {model_path}")
    model = PeftModel.from_pretrained(model, path_checkpoint)
    print(f"peft loaded: {path_checkpoint}")
    model = model.merge_and_unload()
    model.to(torch.bfloat16)
    print("saving ...")
    setattr(model, "_hf_peft_config_loaded", False)
    model.save_pretrained(output_path)
    if tokenizer_path is None:
        print("using model_path as tokenizer path", model_path)
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge lora model with base model')
    parser.add_argument('--model_path', type=str, required=True ,help='path/name to base model')
    parser.add_argument('--lora_path', type=str,  required=True , help='path to lora model')
    parser.add_argument('--output_path', type=str,  required=True ,help='path to output model')
    args = parser.parse_args()

    merge_basic(args.model_path, args.lora_path, args.output_path, tokenizer_path=args.model_path)
    # initiate_model_lora_merge(model_path, lora_path, output_path, merge_weight=1)
    copy_support_files(args.model_path, args.output_path)
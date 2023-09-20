import os

from transformers import AutoModelForCausalLM


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    model.save_pretrained(FILE_PATH + "/../models/test")


if __name__ == "__main__":
    main()

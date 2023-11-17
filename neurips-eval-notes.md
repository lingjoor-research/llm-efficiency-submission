# notes

Overall, this required minimal effort to work.

- Model training artifacts are stored at https://huggingface.co/binaryaaron/neurips-lingjoor-qwen-mix-all-200-v2-1-2-test
- minor modifications made to `Dockerfile` to install required libs to make the API work without other efforts


# running the inference server

- Build and run the `Dockerfile`, not `Dockerfile.train` or `Dockerfile.data`. 

FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the current directory contents into the container at /submission
COPY ./utils/ /submission/utils/
COPY ./scripts/ /submission/scripts/
COPY ./fast_api_requirements.txt /submission/fast_api_requirements.txt
COPY ./main.py /submission/main.py
COPY ./api.py /submission/api.py

# Setup server requriements
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-gpt
RUN apt-get update && apt-get install -y git
RUN pip install huggingface_hub sentencepiece transformers bitsandbytes accelerate scipy

# Load the model and save it to the container
# RUN python scripts/load_and_save_model.py

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

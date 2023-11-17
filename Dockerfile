FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the current directory contents into the container at /submission
COPY ./fast_api_requirements.txt /submission/fast_api_requirements.txt
COPY ./requirements.txt /submission/requirements.txt
COPY ./run_specs.conf /submission/

RUN apt-get update && apt-get install -y git wget curl
# Setup server requriements
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-gpt
RUN pip install --no-cache-dir --upgrade -r requirements.txt 
# adding here to make it work more readily
RUN pip install --no-cache-dir --upgrade crfm-helm 

COPY ./utils/ /submission/utils/
COPY ./main.py /submission/main.py
COPY ./api.py /submission/api.py

# Run the server
# changed to port 8080 to make it work internally
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

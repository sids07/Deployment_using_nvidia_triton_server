FROM nvcr.io/nvidia/tritonserver:23.05-py3 
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y ffmpeg

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir transformers==4.38.2 torch==2.2.1 accelerate==0.22.0 torchaudio
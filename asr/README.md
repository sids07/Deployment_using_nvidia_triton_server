Here, we have only used nvidia-triton-inference-server. But it also support multiple features like TensoRT, vLLM, etc.

Steps to run:

1. cd asr

2. docker build --network host --tag my-tritonserver .

3. docker run --gpus all --shm-size=10G -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/asr:/mnt/model_repository my-tritonserver tritonserver --model-repository=/mnt/model_repository --log-verbose=1


If you want to run and install dependencies without docker then,

1. conda create -k -y -n asr_dependencies python=3.9
2. conda activate asr_dependencies
3. pip install --no-cache-dir transformers==4.38.2 torch==2.2.1 accelerate==0.22.0 torchaudio
4. conda pack -o asr_dependencies.tar.gz

Finally, we need to add parameter key at config.pbtxt

parameters: {
    key: "EXECUTION_ENV_PATH",
    valu: /mnt/model_repository/asr_dependencies.tar.gz
}
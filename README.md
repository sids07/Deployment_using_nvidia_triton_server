# Deployment_using_nvidia_triton_server
Deployed ML models using nvidia-triton

First let's setup Nvidia-Triton Inference server on our device: (https://github.com/triton-inference-server/server)

Easiest way is to use docker. And I prefer using docker for it.

But you can go through their documentation for setting up the server without docker: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/build.html#:~:text=and%20Incremental%20Builds-,Development%20Builds%20Without%20Docker,backend%2C%20or%20a%20repository%20agent. 

For this, I will be using docker. Full steps to follow.

1. They provide us with the image i.e. nvcr.io/nvidia/tritonserver:23.05-py3  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver . Feel free to use the version you want.

2. Now this image might not have all the dependencies you required for running your model so you can create new DockerFile with base image being their image and then we install our dependencies and create new DockerFile. For sample you can find Dockerfile on both asr and llm directory. Then, you will have tritonserver image with your dependencies on it.
    docker build --network host --tag {image_name} .

3. Finally, you need to strictly follow the model directory format which they have provided i.e.
    model_repository #root directory
    --  whisper-large-v3 #model_name
        --  1 ## version
            -- model.py
        --  config.pbtxt

4. Major things to consider in model.py and config.pbtxt:
   a. For model.py, you should strictly use TritonPythonModel as the class name.
   b. Better if you use logger triton server provides for logging. i.e. triton_python_backend_utils.Logger.
   c. For config.pbtxt, name must be exact same as the model_name on the directory format.

5. Finally, after you create config.pbtxt and model.py. You can finally run the docker image to utilize the config and code.
   NOTE: if you are using cloud instance make sure 8000,8001 and 8002 port are exposed as gRPC, https servers runs on those ports.

    docker run --gpus all --shm-size=10G -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/{root_directory_name}:/mnt/model_repository {new_dockerimage_name } tritonserver --model-repository=/mnt/model_repository --log-verbose=1



Major Advantage of using Nvidia-Triton:

1.  Dynamic Batching: We can make concurrent request combine and send request as a batch to our model.
    To activate this feature just add dynamic_batching section to config.pbtxt file.

        dynamic_batching { 
        preferred_batch_size: [2, 4, 8] 
        max_queue_delay_microseconds: 300
        }

    The preferred_batch_size property which indicates the batch sizes that the dynamic batcher should attempt to create.
    The max_queue_delay_microseconds property determines the maximum delay time allowed in the scheduler for other requests to join the dynamic batch.

2. Concurrent Model Execution: Allow us to use multiple GPU efficiently by initiating multiple or same models and loading the required model at that instance and unloading other models.

        instance_group [
            {
            count: 2
            kind: KIND_GPU
            gpus: [ 0 ]
            },
            {
            count: 3
            kind: KIND_GPU
            gpus: [ 1, 2 ]
            }
        ]
    If you have access to multiple GPUs, you can also change the instance_group settings to place multiple execution instances on different GPUs. For example, the above configuration will place two execution instances on GPU 0 and three execution instances on GPUs 1 and 2.

    To make sure that we use GPU efficiently, Triton gives you APIs to load and unload the models via APIs. You can use these controls:

    You can call the POST API  to load the model
    /v2/repository/models/<Model Name>/load

    You can call the POST API  to unload the model
    /v2/repository/models/<Model Name>/unload

    Allowing multiple models to share the GPU memory. This can help optimize memory usage and improve performance.

3. Ensemble Models: An ensemble model represents a pipeline of one or more Machine Learning models whose inputs and outputs are interconnected. This concept can also be applied for the pre and post-processing logic, by treating them as independent blocks/models which are then assembled together on Triton. This approach requires first converting the model into a serialized representation, such as ONNx, before deploying it on the Triton server. 

Code will be available soon.

4. Easy integrate with other tools like vllm, TensoRT, etc.

Code will be available soon.
# Deployment_using_nvidia_triton_server
Deployed ML models using Nvidia-triton

First let's set up the Nvidia-Triton Inference server on our device: (https://github.com/triton-inference-server/server)

The easiest way is to use docker. And I prefer using docker for it.

But you can go through their documentation for setting up the server without docker: 

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/build.html#:~:text=and%20Incremental%20Builds-,Development%20Builds%20Without%20Docker,backend%2C%20or%20a%20repository%20agent. 

For this, I will be using docker. Full steps to follow.

1. They provide us with the image i.e. nvcr.io/nvidia/tritonserver:23.05-py3 (https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver). Feel free to use the version you want.

2. Now this image might not have all the dependencies you require for running your model so you can create a new DockerFile with the base image being their image and then we install our dependencies and create a new DockerFile. For a sample, you can find the Dockerfile in the ASR directory. Then, you will have a tritonserver image with your dependencies on it.
    ```
   docker build --network host --tag {image_name} .
    ```

    ALTERNATIVE:
   You can also install all the dependencies on the conda environment pack them in a tar.gz file and pass that file as a parameter on config.pbtxt

   E.g. for automatic speech recognition using hugging face transformer library below are the dependencies.
   ```
    1. conda create -k -y -n asr_dependencies python=3.9
    2. conda activate asr_dependencies
    3. pip install --no-cache-dir transformers==4.38.2 torch==2.2.1 accelerate==0.22.0 torchaudio
    4. conda pack -o asr_dependencies.tar.gz
   ```

    Finally, we need to add parameter key at config.pbtxt
    ```
    parameters: {
        key: "EXECUTION_ENV_PATH",
        value: /mnt/model_repository/asr_dependencies.tar.gz
    }
   ```
   

4. Finally, you need to strictly follow the model directory format that they have provided i.e.
    model_repository #root directory
    ```
    --  whisper-large-v3 #model_name
        --  1 ## version
            -- model.py
        --  config.pbtxt
    ```

5. Major things to consider in model.py and config.pbtxt:
   ```
   a.For model.py, you should strictly use TritonPythonModel as the class name.
   b. Better if you use logger triton server provides for logging. i.e. triton_python_backend_utils.Logger.
   c. For config.pbtxt, the name must be exactly the same as the model_name on the directory format.
   ```
   
7. Finally, after you create config.pbtxt and model.py. You can finally run the docker image to utilize the config and code.
   NOTE: if you are using a cloud instance make sure 8000,8001 and 8002 ports are exposed as gRPC, https servers run on those ports.

   ```
   docker run --gpus all --shm-size=10G -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/{root_directory_name}:/mnt/model_repository {new_dockerimage_name } tritonserver --model-repository=/mnt/model_repository --log-verbose=1
   ```

Major Advantages of using Nvidia-Triton:

1.  Dynamic Batching: We can make concurrent requests combine and send requests as a batch to our model.
    To activate this feature just add the dynamic_batching section to config.pbtxt file.

        dynamic_batching { 
        preferred_batch_size: [2, 4, 8] 
        max_queue_delay_microseconds: 300
        }

    The preferred_batch_size property which indicates the batch sizes that the dynamic batcher should attempt to create.
    The max_queue_delay_microseconds property determines the maximum delay time allowed in the scheduler for other requests to join the dynamic batch.

2. Concurrent Model Execution: Allows us to use multiple GPUs efficiently by initiating multiple or the same models loading the required model at that instance and unloading other models.

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
    `/v2/repository/models/<Model Name>/load`

    You can call the POST API  to unload the model
    `/v2/repository/models/<Model Name>/unload`

    Allowing multiple models to share the GPU memory. This can help optimize memory usage and improve performance.

3. Ensemble Models: An ensemble model represents a pipeline of one or more Machine Learning models whose inputs and outputs are interconnected. This concept can also be applied to the pre and post-processing logic, by treating them as independent blocks/models which are then assembled together on Triton. This approach requires first converting the model into a serialized representation, such as ONNx, before deploying it on the Triton server. 

For e.g. if we have different modules for the tokenizer and model then we can utilize ensemble model functionality as:
```
ensemble_scheduling {
    step [
        {
            model_name: "tokenizer"
            model_version: -1
            input_map {
            key: "prompt"
            value: "prompt"
        }
        output_map [
        {
            key: "input_ids"
            value: "input_ids"
        },
        {
            key: "attention_mask"
            value: "attention_mask"
        }
        ]
        },
        {
            model_name: "model"
            model_version: -1
        input_map [
            {
                key: "input_ids"
                value: "input_ids"
            },
            {
                key: "attention_mask"
                value: "attention_mask"
            }
        ]
        output_map {
                key: "output_0"
                value: "output_0"
            }
        }
    ]
}
```
Full Code using the ensemble method will be available soon.

4. Easy integration with other tools like vllm, TensoRT, etc.

The full Code for vllm will be available soon.

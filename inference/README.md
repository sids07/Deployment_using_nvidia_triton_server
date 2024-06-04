Here, we learn how we do inference on our nvidia triton inference server.

By default nvidia-triton-inference server gives us both grpc as well as http client.

http client is hosted on 0.0.0.0:8000
grpc client is hosted on 0.0.0.0:8001

Let's start by installing dependencies:
    pip install -r requirements.txt

NOTE: here we do inference to the asr model. And for this usages i have use approach where we pass audio_buffer and sampling rate as input instead of directly passing audio_path. However, we can use audio_path directly as well but we need to make changes accordingly on server and client end.

Code snippet:

    grpc_inference= InferenceClient(inference_type="grpc")
    
    audio_path = "PATH_TO_YOUR_AUDIO_FILE"
    
    triton_model_name = "whisper-large-v3"
    
    output_name = "transcribed_text"
    
    text = grpc_inference.get_response(
        audio_path= audio_path, 
        triton_model_name= triton_model_name, 
        output_name= output_name
    )
    
    print(text)

Lets discuss the code:
1. strictly, inference_type must be either "grpc" or "http".
2. for audio_path, specify the audio which you want to transcribe.
3. triton_model_name, here you need to specify the model name in the config.pbtxt file.
4. output_name, here you need to specify the name of output which you want to see from config.pbtxt file.

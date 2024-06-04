import tritonclient.grpc as grpcclient, httpclient
from scipy.io import wavfile
import numpy as np

class InferenceTypeError(Exception):
    
    def __init__(self, message: str = "Inference Type must be either 'grpc' or 'http'"):
        
        self.message = message
        super().__init__(self.message)


class InferenceClient:
    
    def __init__(self, inference_type :str ="grpc"):
        
        if inference_type not in ["grpc","http"]:
            raise InferenceTypeError()
        
        self.inference_type = inference_type
        self.client, self.inference_client = self.configure_client() 
    
    def configure_client(self):
        
        if self.inference_type == "grpc":
            return self._configure_grpc()
        
        elif self.inference_type == "http":
            return self._configure_http()
        
    def _configure_grpc():
        
        client = grpcclient
        host_url = "0.0.0.0:8001"
        inference_client = client.InferenceServerClient(url = host_url)
        
        return client, inference_client
    
    def _configure_http():
        client = httpclient
        host_url = "0.0.0.0:8000"
        inference_client = client.InferenceServerClient(url = host_url)
        
        return client, inference_client
    
    def get_audio_sampling_rate(self, audio_path):
        
        sampling_rate, audio_buffer = wavfile.read(audio_path)
        
        audio_buffer = audio_buffer.astype(np.float32)
        audio_buffer /= np.max(np.abs(audio_buffer))
        
        return audio_buffer, sampling_rate
    
    def make_input_for_server(self, audio_path):
        inputs = []
        audio_buffer, sampling_rate = self.get_audio_buffer_sampling_rate(audio_path)
        
        inputs.append(self.client.InferInput("audio",[audio_buffer.shape[0],], datatype="FP32"))
        inputs[0].set_data_from_numpy(audio_buffer)
        
        inputs.append(self.client.InferInput("sampling_rate",[1], datatype="INT32"))
        inputs[1].set_data_from_numpy(np.array([sampling_rate]).astype(np.int32))
        
        return inputs
    
    def get_response(self, audio_path, triton_model_name, output_name):
        
        inputs = self.make_input_for_server(audio_path= audio_path)
        
        server_response = self.inference_client.infer(model_name=triton_model_name, inputs=inputs)
        
        return server_response.as_numpy(name=output_name)[0]
    

if __name__ =="__main__":
    
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
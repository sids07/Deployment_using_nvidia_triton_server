import triton_python_backend_utils as pb_utils
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import numpy as np
import time

class TritonPythonModel:
    
    def __init__(
        self,
        device="cuda", 
        model_name="openai/whisper-large-v3"
        ) -> None:
        self.logger = pb_utils.Logger
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype = torch_dtype,
            low_cpu_mem_usage = True,
            use_safetensors = True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(
            model_name
        )
        
        self.transcriber_pipeline = pipeline(
            "automatic-speech-recognition",
            model = model,
            tokenizer = processor.tokenizer,
            feature_extractor = processor.feature_extractor,
            chunk_length_s = 30,
            device = device,
            batch_size = 16,
            torch_dtype = torch_dtype
        )
        
    def execute(self, requests):
        responses = []
        start_time_batch = time.perf_counter()
        for request in requests:
            start_time = time.perf_counter()
            audio_input = pb_utils.get_input_tensor_by_name(request, "audio")
            sampling_rate = pb_utils.get_input_tensor_by_name(request,"sampling_rate")
            
            audio = audio_input.as_numpy()
            sampling_rate = sampling_rate.as_numpy()[0]
            transcribed_text = self.transcriber_pipeline(
                {
                    "sampling_rate": sampling_rate, 
                    "raw": audio
                },
                generate_kwargs={"language":"english"}
            )["text"]
            self.logger.log_info(f"Transcribed Text: {transcribed_text}...")
            inference_response = pb_utils.InferenceResponse(
                output_tensors = [
                    pb_utils.Tensor("transcribed_text", np.array([transcribed_text.encode()]))
                ]
            )
            self.logger.log_info(f"Time taken for single request: {time.perf_counter()- start_time}")
            responses.append(inference_response)
        self.logger.log_info(f"Time taken by batch: {time.perf_counter()- start_time_batch}")
        return responses
    
    def finalize(self, args):
        self.transcriber_pipeline = None
        
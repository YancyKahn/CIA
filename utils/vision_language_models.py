from transformers import (AutoProcessor, 
                          LlavaForConditionalGeneration,
                          Blip2ForConditionalGeneration,
                          InstructBlipProcessor,
                          InstructBlipForConditionalGeneration)

import torch

class VLLM():
    
    def __init__(self, model_name, model_path, device_map="auto"):
        self.model_name = model_name
        self.model_path = model_path
        self.device_map = device_map

    def load_model_and_processor(self):
        if self.model_name == 'llava':
            self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                           use_faset=True,
                                                           trust_remote_code=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
                trust_remote_code=True
            )

        elif self.model_name == 'blip2':
            self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                           use_faset=True,
                                                           trust_remote_code=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
                trust_remote_code=True
            )
        elif self.model_name == "instruct-blip":
            self.processor = InstructBlipProcessor.from_pretrained(self.model_path,
                                                           use_faset=True,
                                                           trust_remote_code=True)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
                trust_remote_code=True
            )
        else:
            raise ValueError('Invalid model name')
        
        return self.processor, self.model
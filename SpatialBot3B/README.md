---
license: cc-by-4.0
datasets:
- RussRobin/SpatialQA
language:
- en
tags:
- Embodied AI
- MLLM
- VLM
- Spatial Understanding
- Phi-2
pipeline_tag: visual-question-answering
---

SpatialBot is a VLM with spatial understanding and reasoning abilties, by precisely understanding depth maps and using them to do high-level tasks.

In this HF repo, we provide the merged SpatialBot-3B, which is based on Phi-2 and SigLIP. It can perform well on general VLM tasks and spatial understanding benchmarks like SpatialBench.

## How to use SpatialBot-3B
### NOTE: We update the repo and quick start codes in 28 August, 2024. Please update your model and codes if you downloaded them before this date.
1. Install dependencies first:
```
pip install torch transformers accelerate pillow numpy
```

2. Run the model:
```
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import numpy as np

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu

model_name = 'RussRobin/SpatialBot-3B'
offset_bos = 0

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

# text prompt
prompt = 'What is the depth value of point <0.5,0.2>? Answer directly from depth map.'
text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(device)

image1 = Image.open('rgb.jpg')
image2 = Image.open('depth.png')

channels = len(image2.getbands())
if channels == 1:
    img = np.array(image2)
    height, width = img.shape
    three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
    three_channel_array[:, :, 0] = (img // 1024) * 4
    three_channel_array[:, :, 1] = (img // 32) * 8
    three_channel_array[:, :, 2] = (img % 32) * 8
    image2 = Image.fromarray(three_channel_array, 'RGB')

image_tensor = model.process_images([image1,image2], model.config).to(dtype=model.dtype, device=device)

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=100,
    use_cache=True,
    repetition_penalty=1.0 # increase this to avoid chattering
)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
```

### Paper: 
https://arxiv.org/abs/2406.13642

### GitHub repo:
https://github.com/BAAI-DCAI/SpatialBot

<!-- ### SpatialQA, the training set:
https://huggingface.co/datasets/RussRobin/SpatialQA
 -->
### SpatialBench, the benchmark:
https://huggingface.co/datasets/RussRobin/SpatialBench

### CKPTs for SpatialBot-3B with LoRA:
https://huggingface.co/RussRobin/SpatialBot-3B-LoRA
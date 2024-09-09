import torch
import json
from diffusers import DiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda:7"

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to(device)

with open("task_1_controlnet/data/test_prompts.json", "r") as f:
    prompts = json.load(f)
    prompts = list(prompts.values())

for i, prompt in enumerate(prompts):
    with torch.autocast("cuda"):
        images = pipe(prompt).images

    for idx, image in enumerate(images):
        image.save(f"task_0_results/{prompt.replace(' ', '_')}.png")

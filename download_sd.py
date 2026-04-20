import huggingface_hub
import sys

# --- THE PATCH MUST GO BEFORE IMPORTING DIFFUSERS ---
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    sys.modules['huggingface_hub'].cached_download = huggingface_hub.hf_hub_download
# ----------------------------------------------------

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"

print("Downloading and loading the model...")
# This automatically downloads the model to your ~/.cache/huggingface folder
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 # Use float16 to save VRAM if you have a GPU
)

print("Download complete!")
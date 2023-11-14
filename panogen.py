import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler


# Load the pre-trained model and initialize the pipeline (as you did in your code)
model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
    model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
)

# Move the pipeline to GPU
# pipe = pipe.to("cuda")

prompt = input("Enter Your Image Prompt: ")

image = pipe(prompt).images[0]

image.save('static/generatedpanorama.png')

# show image
from PIL import Image

#read the image
im = Image.open("static/generatedpanorama.png")

#show image
im.show()
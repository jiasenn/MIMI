import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
    model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
    )

pipe = pipe.to("cuda")

prompt = "Singapore river at night"
image = pipe(prompt).images[0]

# save image
image.save('generatedpanorama.png')

# display image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()

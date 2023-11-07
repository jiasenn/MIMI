# from flask import Flask, render_template, request, redirect, url_for

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/result')
# def display_result():
#     return render_template('result.html')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, send_file
import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

app = Flask(__name__)

# Load the pre-trained model and initialize the pipeline (as you did in your code)
model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
    model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
)

# Move the pipeline to GPU
pipe = pipe.to("cuda")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def display_result():
    # Get the prompt from the form
    prompt = "Singapore river at night"
    # Generate the image
    image = pipe(prompt).images[0]

    # Save the generated image
    image.save('static/generatedpanorama.png')
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)

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

from flask import Flask, render_template, request, redirect, url_for, jsonify
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

# @app.route('/result', methods=['POST'])

@app.route('/send-data-to-server', methods=['POST'])
def receive_data():
    answers = ""
    # Receive the data sent from the client (JavaScript)
    data = request.get_json()
    print(data)
    # Process the data as needed
    # For example, you can access it like this:
    # append each answer to the answers list
    # print("Question:", data[1])
    # print("Option:", data[-2])
    
    # answer = str("Question:" + data[1] + " Option:" + data[-2] + "\n")
    # answers = answers + answer
    # answers.append(answer)
    # Perform some processing or return a response
    response_data = {
        "message": "Data received and processed successfully",
    }
    # split the query parameter into a list of strings
    data = data.split('&')
    quiz_answers = ""

    for qn in range(len(data) - 2):
        quiz_answers = quiz_answers + "Question " + str(qn+1) + ": Option: " + data[qn][-1] + "\n"
        

    # save quiz_answers to a file
    with open('static/quiz_answers.txt', 'w') as f:
        f.write(quiz_answers)
    print("Successfully saved quiz answers to file")
    # return the data as a JSON response
    return jsonify(response_data)
    

@app.route('/result', methods=['GET'])
def display_result():
    with open('static/quiz_answers.txt', 'r') as f:
        answers_lst = f.readlines()

    # Generate an image based on the answers
    # prompt = " ".join(quiz_answers)  # Combine all answers into a single prompt
    # prompt = "Fort Canning Park in Singapore with people walking around"
    # image = pipe(prompt).images[0]

    # Save the generated image
    # image.save('static/generatedpanorama.png')
    print(answers_lst)

    # display the quiz answers on the result page
    return render_template('result.html', quiz_answers=answers_lst)

if __name__ == '__main__':
    app.run(debug=True)


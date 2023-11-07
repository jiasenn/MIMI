from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler
import logging
import sys
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Change INFO to DEBUG if you want more extensive logging
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

app = Flask(__name__)

# Load the pre-trained LlamaIndex model and initialize the pipeline
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

llm = LlamaCPP(
    model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf",
    
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    
    temperature=0.0,
    max_new_tokens=1024,
    
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
    
    # kwargs to pass to __call__()
    generate_kwargs={},
    
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 30}, # I need to play with this and see if it actually helps
    
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# Create an index of your documents
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
import os

storage_directory = "./storage"

documents = SimpleDirectoryReader('./testdata').load_data()

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024,
                                               embed_model="local",
                                               callback_manager=callback_manager)

storage_context = StorageContext.from_defaults(persist_dir=storage_directory)

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
    # Receive the data sent from the client (JavaScript)
    data = request.get_json()
    response_data = {
        "message": "Data received and processed successfully",
    }
    # split the query parameter into a list of strings
    data = data.split('&')
    quiz_answers = ""

    for qn in range(len(data) - 2):
        quiz_answers = quiz_answers + "Question " + str(qn+1) + ": Option: " + data[qn][-1] + "\n"
        
    # save quiz_answers to a file
    with open('testdata/choices.txt', 'w') as f:
        f.write(quiz_answers)
    print("Successfully saved quiz answers to file")
    # return the data as a JSON response
    return jsonify(response_data)
    

@app.route('/result', methods=['GET'])
def display_result():
    with open('testdata/choices.txt', 'r') as f:
        answers_lst = f.readlines()

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # Persist the index to disk
    index.storage_context.persist(persist_dir=storage_directory)
    query_engine = index.as_query_engine(service_context=service_context,
                                        similarity_top_k=3)

    query = "Given the quiz from quiz.txt and the results from choices.txt describe my personality and the archetype from the information in all the files provided. \n" \
            "Personality should not be an archetype. \n" \
            "Please summarize your reasoning into 3 sentences. \n" \
            "Please use the following format: \n" \
            "Your personality is <personality> because <reasoning>. \n" \
            "Your archetype is <archetype> because <reasoning>. \n" 
            

    img_prompt = "Given the quiz from quiz.txt and the results from choices.txt describe my personality and the archetype from the information in all the files provided. \n" \
                "What is your archetype and which location in Singapore best represents this archetype? \n" \
                "Please summarize your reasoning into 1 sentences using the following format to summarize: \n" \
                "<archetype> in location such as <location>."
    
    # print time taken to generate response
    import time
    t_start = time.time()
    start = time.time()
    description = query_engine.query(img_prompt)
    description = str(description)
    end = time.time()
    
    print("Time taken to generate description: ", end - start)

    # r_start = time.time()
    # prompt = query_engine.query(img_prompt)
    # # convert prompt to a string
    # prompt = str(prompt)
    # r_end = time.time()
    # print("Time taken to generate response: ", r_end - r_start)

    print("Description: ", description)
    # print("Response:", prompt)

    image = pipe(description).images[0]
    # Save the generated image
    image.save('static/generatedpanorama.png')
    t_end = time.time()

    print("Time taken to generate image: ", t_end - t_start)
    # display the quiz answers on the result page
    return render_template('result.html', quiz_answers=description)
    # return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)


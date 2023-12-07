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
from PIL import Image 
import PIL  

app = Flask(__name__)

# Load the pre-trained LlamaIndex model and initialize the pipeline
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

llm = LlamaCPP(
    model_url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf",
    
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    # model_path="/Users/jiasenn/Downloads/llama-2-13b-chat.Q5_K_M.gguf",
    
    temperature=0.0,
    max_new_tokens=1024,
    
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,  # note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
    
    # kwargs to pass to __call__()
    generate_kwargs={},
    
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 10}, # I need to play with this and see if it actually helps
    
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

pipe.circular_padding = True

# from diffusers import DiffusionPipeline
# pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=False)

# from diffusers import StableDiffusionPipelinez
# pipe = DiffusionPipeline.from_pretrained("/home/users/jiasen_chen/scratch/model/more_details.safetensors",torch_dtype=torch.float16,use_safetensors=True)


# Move the pipeline to GPU
pipe = pipe.to("cuda")


@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/index.html')
def index():
    return render_template('index.html')

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
    archetype_count = [0, 0, 0, 0]
    archetype_lst = ["The Tradesman", "The Fisherman", "The Samsui Woman", "The Coolie"]

    for qn in range(len(data) - 2):
        quiz_answers = quiz_answers + "Question " + str(qn+1) + ": Option: " + data[qn][-1] + "\n"
        # print(type(data[qn][-1]))
        if data[qn][-1] == "1":
            # print("hello")
            archetype_count[0] += 1 
            # archetype["The Colonial Administrator"] += 1
        elif data[qn][-1] == "2":
            # archetype["Samsui Women"] += 1
            archetype_count[1] += 1 
        elif data[qn][-1] == "3":
            # archetype["Fisherman"] += 1
            archetype_count[2] += 1 
        elif data[qn][-1] == "4":
            archetype_count[3] += 1 
            # archetype["Tradesman"] += 1        
    
    print(archetype_count)
    global result_archetype
    result_archetype = archetype_lst[archetype_count.index(max(archetype_count))]
    
    # hardcoded LoRa generated archetype image
    if result_archetype == "The Coolie":
        arch_im = Image.open(r"static/Icons/coolie.jpg")
    elif result_archetype == "The Tradesman":
        arch_im = Image.open(r"static/Icons/tradesman.jpeg")
    elif result_archetype == "The Fisherman":
        arch_im = Image.open(r"static/Icons/fisherman.jpeg")
    else:
        arch_im = Image.open(r"static/Icons/samsui.jpeg")
    arch_im.save("static/Icons/genereatedarche.jpg")
    
    print(result_archetype, "is the result archetype")
        
    # save quiz_answers to a file
    with open('testdata/choices2.txt', 'w') as f:
        f.write(quiz_answers)
    with open('testdata/archetype.txt', 'w') as f:
        f.write(result_archetype)
    print("Successfully saved quiz answers to file")
    # return the data as a JSON response
    return jsonify(response_data)
    

@app.route('/result', methods=['GET'])
def display_result():
    # with open('testdata/choices2.txt', 'r') as f:
    #     answers_lst = f.readlines()
    
    storage_directory = "./storage"

    documents = SimpleDirectoryReader('./testdata').load_data()

    # storage_context = StorageContext.from_defaults(persist_dir=storage_directory)

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # Persist the index to disk
    index.storage_context.persist(persist_dir=storage_directory)
    query_engine = index.as_query_engine(service_context=service_context,
                                        similarity_top_k=3)

    # query = "Given the quiz from quiz2.txt, the results from choices2.txt and archetype from archetype.txt, describe my personality from the information in all the files provided. \n" \
    #         "Personality should not be an archetype. \n" \
    #         "Please summarize your reasoning into 3 sentences. \n"
            # "Please use the following format: \n" \
            # "Your personality is <personality> because <reasoning>. \n" \
            # "Your archetype is <archetype> because <reasoning>. \n" 
            
    qn = "What is the archetype in archetype.txt. Describe user personalities from the quiz answers in choices2.txt. \n" \
        "Dont mention other archetype not in archetype.txt. \n" \
        "Please use the following format: \n" \
        "Based on the quiz you are <personality> because <reasoning>. \n"

    # img_prompt = "Given the quiz from quiz2.txt and the results from choices2.txt describe my personality and the archetype from the information in all the files provided. \n" \
    #             "What is your archetype and which location in Singapore best represents this archetype? \n" \
    #             "Please summarize your reasoning into 1 sentences using the following format to summarize: \n" \
    #             "<archetype> in location such as <location>."
    
    # a_prompt = "My archetype is? \n" \
    #             "Choose one of the following: \n" \
    #             "The Tradesman \n" \
    #             "The Fisherman \n" \
    #             "The Samsui Woman \n" \
    #             "The Coolie"

    # l_prompt = "The location that best represents my archetype is?"
    
    # print time taken to generate response
    import time
    t_start = time.time()
    start = time.time()
    # description = query_engine.query(img_prompt)
    # description = str(description)
    # result_archetype = query_engine.query(a_prompt)
    # result_archetype = str(result_archetype)
    # location = query_engine.query(l_prompt)

    locations = ["Clarke Quay", "Fort Canning Park", "Raffles Hotel", "Chinatown", "Singapore River", "Bugis Street", "Little India"]
    # randomly choose 1 location from the list of locations
    import random
    location = str(random.choice(locations))
    # print("Archetype: ", result_archetype)
    print("Location: ", location)
    

    # r_start = time.time()
    # prompt = query_engine.query(img_prompt)
    # # convert prompt to a string
    # prompt = str(prompt)
    # r_end = time.time()
    # print("Time taken to generate response: ", r_end - r_start)
    with open('testdata/archetype.txt', 'r') as f:
        result_archetype = f.read()
    
    result_archetype = str(result_archetype)
    print(result_archetype)
    prompt = location + "Singapore in the 1980s, 360 image."
    # description = "Your archetype is " + result_archetype + " because " + location + " is a place where " + result_archetype + " can be found."
    description = query_engine.query(qn)

    # hardcoded description
    # if result_archetype == "The Coolie":
    #     description = "Your choices show a celebration of cultural diversity and a recognition of the struggles and stories of the past. This aligns with the experiences of the coolies and immigrant laborers, who contributed immensely to the multicultural tapestry of Singapore."
    # elif result_archetype == "The Tradesman":
    #     description = "Your selections suggest you have a knack for innovation and an interest in the bustling world of business and technology. You are forward-thinking, much like the visionary traders and merchants who played a key role in Singapore's history."
    # elif result_archetype == "The Fisherman":
    #     description = "Your choices reflect a deep appreciation for nature, tranquility, and the simpler things in life. This mirrors the spirit of the fishermen and villagers, who were closely tied to the natural beauty and serenity of early Singapore."
    # else:
    #     description = "You resonate with the qualities of hard work, dedication, and community spirit, similar to the Samsui women. These qualities were essential in shaping Singapore's development, reflecting a strong commitment to collective progress."

    description = str(description)

    end = time.time()
    
    print("Time taken to generate description: ", end - start)

    print("Archetype: ", result_archetype)
    print("Description: ", description)

    
    # print("Response:", prompt)

    image = pipe(prompt).images[0]
    # Save the generated image
    image.save('static/Icons/generatedpanorama.jpg')
    t_end = time.time()

    print("Time taken to generate image: ", t_end - t_start)
    # display the quiz answers on the result page
    return render_template('result.html', quiz_answers=description, result_archetype=result_archetype)
    # return render_template('result.html')

@app.route('/explore.html')
def explore():
    return render_template('explore.html')

@app.route('/collection.html')
def collection():
    return render_template('collection.html')

@app.route('/profile.html')
def profile():
    return render_template('profile.html')

# hardcoded result
# @app.route('/profile.html')
# def profile():
#     with open('testdata/archetype.txt', 'r') as f:
#         result_archetype = f.read()
    
#     result_archetype = str(result_archetype)
#     print(result_archetype)
#     return render_template('profile.html', result_archetype=result_archetype)


if __name__ == '__main__':
    app.run(debug=True)


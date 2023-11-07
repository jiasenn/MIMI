import tkinter as tk
import customtkinter as ctk 

from PIL import ImageTk
from apikey import apikey as auth_token

import torch
print(torch.__version__)
from torch import autocast
from diffusers import StableDiffusionPipeline 

import logging
import sys

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Change INFO to DEBUG if you want more extensive logging
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index import StorageContext, load_index_from_storage

storage_directory = "./storage"

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=1024,
                                                embed_model="local",
                                                callback_manager=callback_manager)

if storage_directory is not None:
    storage_context = StorageContext.from_defaults(persist_dir=storage_directory)
    index = load_index_from_storage(storage_context, service_context=service_context)

else:
    documents = SimpleDirectoryReader('./testdata').load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # Persist the index to disk
    index.storage_context.persist(persist_dir=storage_directory)

# Query your index!
from IPython.display import Markdown, display
from llama_index.prompts import PromptTemplate

query_engine = index.as_query_engine(service_context=service_context,
                                     similarity_top_k=3)

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("MIMI Generator") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
    
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

response = query_engine.query("From the files loaded, what is the name of the file talks about hard labour?")
print(response)

def generate(): 
    with autocast(device): 
        # prompt = index.query("hello", k=1)[0]
        image = pipe(prompt.get(), guidance_scale=8.5)["images"][0]
        # print(pipe(prompt.get(), guidance_scale=8.5).keys())

    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()
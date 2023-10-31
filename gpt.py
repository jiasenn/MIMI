import os, streamlit as st
from apikey import apikey as auth_token


# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
os.environ['OPENAI_API_KEY'] = auth_token

from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI
import openai


# Define a simple Streamlit app
st.title("Ask Llama")
query = st.text_input("What would you like to ask? (source: data/paul_graham_essay.txt)", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            openai.api_key = auth_token

            # This example uses text-davinci-003 by default; feel free to change if desired
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

            # Configure prompt parameters and initialise helper
            max_input_size = 4096
            num_output = 256
            max_chunk_overlap = 20

            prompt_helper = PromptHelper(max_input_size, num_output, chunk_overlap_ratio= 0.1, chunk_size_limit=max_chunk_overlap)

            # Load documents from the 'data' directory
            documents = SimpleDirectoryReader('./testdata').load_data()
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            
            response = index.query(query)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
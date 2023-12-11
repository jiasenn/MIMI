# Define the archetype descriptions and scoring system
archetypes = {
    "Colonial Administrator": 0,
    "Coolie/Immigrant Labourer": 0,
    "Samsui Woman": 0,
    "Fisherman/Villager": 0,
    "Freedom Fighter": 0,
    "Pioneer Politician": 0,
    "Straits Chinese": 0,
    "Kampong Leader": 0,
    "Tradesman/Merchant": 0,
    "Educator": 0,
}

# Define the quiz questions
questions = [
    # Question 1
    "Morning Routine: How do you start your day in CityScape?",
    "   1. Checking news and updates for the city.",
    "   2. Heading straight to work, no time to lose.",
    "   3. Taking care of family or neighbors.",
    "   4. Visiting a community garden or local park.",
    "   5. Participating in a community rally or cause.",
    # Question 2
    "Preferred Transport: How do you move around CityScape?",
    "   1. Chauffeured car or private cab.",
    "   2. Public transport, mingling with the crowds.",
    "   3. Walking, while helping others along the way.",
    "   4. Bicycle or eco-friendly options.",
    "   5. Organizing carpools or group rides.",
    # Question 3
    "City Event: Which event are you most likely to attend?",
    "    1. A leadership summit or conference.",
    "    2. A cultural festival celebrating heritage.",
    "    3. A community mentorship program.",
    "    4. A trade fair or business expo.",
    "    5. An educational seminar or workshop.",
    # Question 4
    "Weekend Activity: How do you spend your weekends in CityScape?",
    "    1. Planning for the city's growth and development.",
    "    2. Exploring cultural districts and traditions.",
    "    3. Organizing or attending community meetings.",
    "    4. Checking out new business opportunities.",
    "    5. Volunteering or teaching.",
    # Question 5
    "Night Out: Your idea of a perfect night out in CityScape?",
    "    1. An elite gathering or gala.",
    "    2. A vibrant street food market.",
    "    3. A family or community dinner.",
    "    4. A lakeside picnic or nature walk.",
    "    5. Attending a social activism event.",
    # Question 6
    "City Issue: CityScape faces a challenge. How do you tackle it?",
    "    1. Gather leaders and strategize.",
    "    2. Collaborate with communities for solutions.",
    "    3. Offer guidance and wisdom to those affected.",
    "    4. Invest resources and find economic solutions.",
    "    5. Educate the public on the issue.",
    # Question 7
    "Vacation Spot: Where would you unwind outside CityScape?",
    "    1. A luxurious retreat.",
    "    2. A backpacking adventure.",
    "    3. A family-friendly resort.",
    "    4. A nature-based getaway.",
    "    5. A historical or educational tour.",
    # Question 8
    "6City Legacy: What do you wish to contribute to CityScape?",
    "    1. Visionary leadership and governance.",
    "    2. Preservation of culture and traditions.",
    "    3. Community bonding and mentorship.",
    "    4. Economic growth and business opportunities.",
    "    5. Knowledge and education for all."
]

# Function to calculate the archetype result
def calculate_result(archetypes):
    max_archetype = max(archetypes, key=archetypes.get)
    return max_archetype

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



# Start the quiz
print("Welcome to CityScape Singapore - Discover Your Archetype!\n")
print("Answer the following questions to find out which historical archetype resonates with your personality:")
choices = []
for i in range(0, len(questions), 6):
    print("\nQuestion " + str(int(i/6) + 1))
    print("\n".join(questions[i:i+6]))
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if choice < 1 or choice > 5:
                print("Please enter a number between 1 and 5.")
            else:
                break  # Break out of the loop if a valid choice is provided
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    choices.append("Question " + str(int(i/6) + 1) + ": Option " + str(choice))
    archetypes[list(archetypes.keys())[choice - 1]] += 1

# Calculate and display the result
result = calculate_result(archetypes)
choices.append("Archetype: " + result)

print("\nYour result is: " + result)

# Sace choices to a file
with open("./data/choices.txt", "w") as file:
    for choice in choices:
        file.write(str(choice) + "\n")
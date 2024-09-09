import random
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import requests
import io

# import token from .env
import os
from dotenv import load_dotenv
load_dotenv()




API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {os.getenv('HF_token')}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

def generate_image(image_desc):
    image_bytes = query({
        "inputs": image_desc,
    })
    image = Image.open(io.BytesIO(image_bytes))
    return image

luck_map = {
    0: "very unfortunate (an event that physically or mentally harms a lot the protagonist)",
    20: "unfortunate (an event that physically or mentally harms the protagonist)",
    40: "neutral (an event that does not affect the protagonist in any way)",
    60: "lucky (an event that benefits the protagonist)",
    80: "very lucky (an event that greatly benefits the protagonist)",
    100: "extremely lucky (an event that greatly benefits the protagonist)",

}

api_key=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.6)
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.1:latest")
initial_choice = "science fiction"

# chains
# --------
setup_prompt = ChatPromptTemplate.from_template("""You are tasked with creating a very brief setup for a story based on the theme of {theme}, about a {gender}. The setup should include the following elements:

- Main Character: Describe the protagonist’s key characteristics, including their personality, background, abilities, and any defining physical traits. very briefly.

- First Situation: Describe the first event, the initial event of the story that sets the plot in motion (a very simple signle event and not a succession of events). This could be a chance encounter, a sudden crisis, or a mysterious discovery that leads the protagonist on a journey or quest. Ensure the situation is engaging and hooks the reader’s interest yet simple and NOT COMPLEX, and that is generalist enough to create many events from it.

Examples of outputs:
- Main Character: You are a exobiologist who has been studying alien life forms for years. You are known for your keen intellect and your ability to think on your feet. You have a prosthetic arm that you designed yourself, which gives you an edge in the field.
- First Situation: One day, while exploring a distant planet, you stumble upon a strange alien artifact that emits a mysterious energy. As you investigate further, you realize that the artifact is a key to unlocking the secrets of the universe, and you are the only one who can unlock its power.""")

setup_chain = setup_prompt | llm | StrOutputParser()
# --------
initial_prompt = ChatPromptTemplate.from_template("""Having the following setup of a story:
{setup}
Imagine a first event that could happen to the protagonist that is in relation with the First Situation described, the event should be different from the First Situation and is a neutral event. The output should be a single sentence.
First Event:""")
initial_chain = initial_prompt | llm | StrOutputParser()
# --------
image_prompt = ChatPromptTemplate.from_template("""Extract one descriptive sentence from the following context:
The context: {context}
Examples of descriptive sentences:
- A dark, stormy night in a deserted town with a lone figure walking down the empty street.
- A lush, vibrant forest filled with exotic plants and colorful creatures.
- An eerie, abandoned mansion with overgrown gardens and broken windows.
- A man in a tattered cloak, standing on a cliff overlooking a vast, misty valley.
- A futuristic cityscape with towering skyscrapers and flying cars zipping through the air.""")
image_chain = image_prompt | llm | StrOutputParser()
# --------
event_prompt = ChatPromptTemplate.from_template("""Having the following setup of a story:
{setup}
Having the following current event happening to the protagonist:
{current_event}
Imagine 3 (three) possible actions the protagonist can do in that situation, the events should be different from each other and can lead to different outcomes, the first possible action is easy, the second possible action is medium in difficulty and the third possible action is hard. Each action should be a single sentence.
The output should be formatted in a numbered list and each element is seperated by a \\n
Examples:
1. Aldin decides to explore the mysterious cave.
2. Eval chooses to follow the strange figure.
3. Abigail opt to return to your ship.
List of actions:""")
event_chain = event_prompt | llm | StrOutputParser()
# ------- 
result_prompt = ChatPromptTemplate.from_template("""Having the following setup of a story:
{setup}
Having the following current event happening to the protagonist:
{current_event}
Having the following action chosen by the protagonist after the current event:
{action}
Imagine a possible {luck_value} event resulted of the action of the protagonist, the event should be different from the current event and can lead to different outcomes. The output should be a single sentence.
Result:""")
result_chain = result_prompt | llm | StrOutputParser()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# Define the game state
class GameState(TypedDict):
    events: list
    theme: str
    gender: str
    hp: int
    mana: int
    luck: int
    setup: str
    current_event: str

def initial_event_node(state: GameState):
    setup = setup_chain.invoke({"theme": state["theme"], "gender": state["gender"]})  # Example theme and gender
    first_event = initial_chain.invoke({"setup": setup})
    state["setup"] = setup
    state["current_event"] = first_event
    state["events"].append(first_event)
    return state

def generate_actions(state: GameState):
    actions = event_chain.invoke({"setup": state["setup"], "current_event": state["current_event"]})
    actions_list = actions.split("\n")
    return state, actions_list

def generate_result(state: GameState, action: str):
    luck = state["luck"]
    luck_value = luck_map[min(luck_map.keys(), key=lambda x:abs(x-luck))]
    result_event = result_chain.invoke({
        "setup": state["setup"],
        "current_event": state["current_event"],
        "action": action,
        "luck_value": luck_value
    })
    state["current_event"] = result_event
    state["events"].append(result_event)
    return state


def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def render_bottom_bar():
    # Icons for bottom bar
    heart_icon = Image.open("./images/heart.png")
    mana_icon = Image.open("./images/mana.png")
    luck_icon = Image.open("./images/luck.png")
    player_icon = Image.open("./images/5804911.png")

    heart_icon_base64 = get_image_base64(heart_icon)
    mana_icon_base64 = get_image_base64(mana_icon)
    luck_icon_base64 = get_image_base64(luck_icon)
    player_icon_base64 = get_image_base64(player_icon)

    # Bottom bar with stats and icons
    st.markdown(f"""
        <div class="bottom-bar">
            <div class="stat-container">
                <img src="data:image/png;base64,{heart_icon_base64}" class="icon">
                <strong>HP</strong><br>
                <progress class="hp" value="{st.session_state.state["hp"]}" max="100"></progress>
            </div>
            <div class="stat-container">
                <img src="data:image/png;base64,{mana_icon_base64}" class="icon">
                <strong>Mana</strong><br>
                <progress class="mana" value="{st.session_state.state["mana"]}" max="100"></progress>
            </div>
            <div class="stat-container">
                <img src="data:image/png;base64,{luck_icon_base64}" class="icon">
                <strong>Luck</strong><br>
                <progress class="luck" value="{st.session_state.state["luck"]}" max="100"></progress>
            </div>
            <div class="stat-container">
                <strong>Player</strong><br>
                <img src="data:image/png;base64,{player_icon_base64}" width="50">
            </div>
        </div>
    """, unsafe_allow_html=True)

def run():
    st.set_page_config(
        page_title="Interactive Story Game",
        page_icon="👾",
        layout="wide"
    )

    st.write("# Welcome to the LLM Story Game👾")

    if 'initial_event' not in st.session_state:
        st.session_state.initial_event = False
        st.session_state.state = {
            "events": [],
            "theme": "",
            "gender": "",
            "hp": 100,
            "mana": 100,
            "luck": 0,
            "setup": "",
            "current_event": ""
        }

    # Custom CSS for bottom bar
    st.markdown("""
        <style>
        .bottom-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f0f0f0;
            padding: 10px;
            text-align: center;
            box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
            z-index: 9999; /* Ensures the bottom bar is always on top */
        }
        .stat-container {
            display: inline-block;
            margin: 0 30px;
        }
        img.icon {
            height: 20px;
            vertical-align: middle;
            margin-right: 10px;
        }
        progress {
            width: 100px;
            height: 20px;
            border-radius: 10px;
            background-color: #eee;
            overflow: hidden;
        }
        progress.hp::-moz-progress-bar,
        progress.hp::-ms-fill {
            background-color: green;
            border-radius: 10px;
        }
        progress.mana::-moz-progress-bar,
        progress.mana::-ms-fill {
            background-color: blue;
            border-radius: 10px;
        }
        progress.luck::-moz-progress-bar,
        progress.luck::-ms-fill {
            background-color: orange;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
        # Form 1: Initial Character Setup

    render_bottom_bar()
    with st.form("character_form"):
        theme_options = ['Fantasy', 'Science Fiction', 'Mystery', 'Adventure', 'Horror']
        theme = st.selectbox("Choose a theme", options=theme_options, index=0)
        custom_theme = st.text_input("Or write your own theme")
        gender = st.radio("Select your gender", ['Man', 'Woman'])
        submitted = st.form_submit_button("Start the Game 🎮")

        if submitted:
            if custom_theme:
                theme = custom_theme

            # Initial state setup
            initial_state = {
                "events": [], 
                "theme": theme, 
                "gender": gender,
                "hp": 100,
                "mana": 100,
                "luck": 0,
                "setup": "",
                "current_event": ""
            }
            initial_event = initial_event_node(initial_state)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # "Your Story Begins" header part
                st.markdown(f"""
                    <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                        <h2 style="color: #4B0082; text-align: center;">Your Story Begins...</h2>
                    </div>
                """, unsafe_allow_html=True)

                # Beautified setup content with a box and some padding
                st.markdown(f"""
                    <div style="padding: 20px; background-color: #ffffff; border-radius: 10px; border: 2px solid #4B0082; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                        <p style="font-size: 16px; color: #333;">
                            {initial_event['setup'].replace('- ', '<br>• ')}
                        </p>
                    </div>
                """, unsafe_allow_html=True)


            with col2:
                with st.spinner("Generating image..."):
                    try:
                        print("image")
                        image = generate_image(initial_event["current_event"])
                        st.image(image, use_column_width=True)
                    except Exception as e:
                        st.error(f"An error occurred while generating the image: {str(e)}")
            st.session_state.initial_event = True
    
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "event_counter" not in st.session_state:
        st.session_state.event_counter = 0

    if st.session_state.initial_event:
        
        # Action form outside the character form
        while st.session_state.state["hp"] > 0:
            print(st.session_state.submitted)
            if not st.session_state.submitted:
                print("not submitted")
                print("event_counter", st.session_state.event_counter)
                with st.form(f"action_form_{st.session_state.event_counter}"):
                    # Generate actions
                    st.session_state.state, actions_list = generate_actions(st.session_state.state)
                    actions_list = [f"{action} ({cost} mana)" for action, cost in zip(actions_list, [0, 10, 20])]
                    print(actions_list, type(actions_list))
                    st.write("Choose an action to be performed by your protagonist:")
                    selected_action = st.radio("Choose an action to be performed by your protagonist:", options = actions_list)

                    submitted_action = st.form_submit_button("Submit action 🚀")
                    
                    if submitted_action:
                        # Save the selected action in session state or print it for debugging
                        print(f"Selected action: {selected_action}")
                        if selected_action == actions_list[0]:
                            mana_cost = 0
                            luck_value = random.choice([0, 20, 40])
                        elif selected_action == actions_list[1]:
                            mana_cost = 10
                            luck_value = random.choice([40, 60, 80])
                        elif selected_action == actions_list[2]:
                            mana_cost = 20
                            luck_value = random.choice([60, 80, 100])

                        st.session_state.state["mana"] -= mana_cost
                        st.session_state.state["luck"] = luck_value

                        if luck_value < 0:
                            st.session_state.state["hp"] -= 50
                        elif luck_value < 50:
                            st.session_state.state["hp"] -= 30
                        elif luck_value >= 50 and luck_value < 80:
                            st.session_state.state["luck"] += random.choice([1, 2, 3, 4, 5, 6, 7])
                        else:
                            st.session_state.state["luck"] += 10

                        st.session_state.state = generate_result(st.session_state.state, selected_action)

                        st.session_state.submitted = True  # Mark the form as submitted
                    else: 
                        st.stop()
            if st.session_state.submitted:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"""
                        <div style="padding: 20px; background-color: #f9f9f9; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                            <h2 style="color: #4B0082; text-align: center;">Current Event</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"{st.session_state.state['current_event']}")

                with col2:
                    with st.spinner("Generating image..."):
                        image = generate_image(st.session_state.state["current_event"])
                        print("image")
                    st.image(image, use_column_width=True)
                st.session_state.event_counter += 1  # Increment event counter
                st.session_state.submitted = False 
            render_bottom_bar()

    render_bottom_bar()

if __name__ == "__main__":
    run()



# def run():
#     st.set_page_config(
#         page_title="Interactive Story Game",
#         page_icon="👾",
#         layout="wide"
#     )

#     st.write("# Welcome to the Story Game")

#     if 'initial_event' not in st.session_state:
#         st.session_state.initial_event = False
#         st.session_state.state = {
#             "events": [],
#             "theme": "",
#             "gender": "",
#             "hp": 100,
#             "mana": 100,
#             "luck": 0,
#             "setup": "",
#             "current_event": ""
#         }


#     # Form 1: Initial Character Setup
#     with st.form("character_form"):
#         theme_options = ['Fantasy', 'Science Fiction', 'Mystery', 'Adventure', 'Horror']
#         theme = st.selectbox("Choose a theme", options=theme_options, index=0)
#         custom_theme = st.text_input("Or write your own theme")
#         gender = st.radio("Select your gender", ['Man', 'Woman'])
#         submitted = st.form_submit_button("Start the Game 🎮")

#         if submitted:
#             if custom_theme:
#                 theme = custom_theme

#             # Initial state setup
#             st.session_state.state = {
#                 "events": [], 
#                 "theme": theme, 
#                 "gender": gender,
#                 "hp": 100,
#                 "mana": 100,
#                 "luck": 0,
#                 "setup": "",
#                 "current_event": ""
#             }
#             st.session_state.state = initial_event_node(st.session_state.state)
#             st.session_state.initial_event = True

#     if st.session_state.initial_event:
#         col1, col2 = st.columns([2, 1])

#         with col1:
#             st.markdown(f"**Your story begins...**")
#             st.markdown(f"{st.session_state.state['setup']}")

#         with col2:
#             st.write("Generating image...")
#             image = generate_image(st.session_state.state["current_event"])
#             st.image(image, use_column_width=True)

#         # Main Game Loop (while hp > 0)
#         while st.session_state.state["hp"] > 0:
#             with st.form("action_form"):
#                 # Generate and display 3 possible actions
#                 st.session_state.state, actions_list = generate_actions(st.session_state.state)
#                 action = st.radio("Choose an action", actions_list)
#                 submitted_action = st.form_submit_button("Submit Action 🚀")

#                 if submitted_action:
#                     # Assign Mana and Luck based on chosen action
#                     mana_cost, luck_value = 0, 0
#                     if action == actions_list[0]:
#                         mana_cost = 0
#                         luck_value = -20  # Unfortunate event
#                     elif action == actions_list[1]:
#                         mana_cost = 10
#                         luck_value = 50  # Moderate luck event
#                     elif action == actions_list[2]:
#                         mana_cost = 20
#                         luck_value = 100  # Lucky event

#                     # Update stats
#                     st.session_state.state["mana"] -= mana_cost
#                     st.session_state.state["luck"] = luck_value

#                     # Determine event outcome based on luck and update HP
#                     if luck_value < 0:
#                         st.session_state.state["hp"] -= 20
#                     elif luck_value < 50:
#                         st.session_state.state["hp"] -= 10
#                     else:
#                         st.session_state.state["hp"] -= 0  # Lucky event, no HP loss

#                     # Generate result of the action
#                     st.session_state.state = generate_result(st.session_state.state, action)

#                     # Display the result and updated stats
#                     col1, col2 = st.columns([2, 1])
#                     with col1:
#                         st.markdown(f"**Current Event:** {st.session_state.state['current_event']}")

#                     with col2:
#                         st.write("Generating image...")
#                         image = generate_image(st.session_state.state["current_event"])
#                         st.image(image, use_column_width=True)

#                     st.write(f"**HP:** {st.session_state.state['hp']} | **Mana:** {st.session_state.state['mana']} | **Luck:** {st.session_state.state['luck']}")

#             # Exit loop if HP reaches 0
#             if st.session_state.state["hp"] <= 0:
#                 st.write("**Game Over!** Your protagonist could not survive the journey.")
#                 break

#     # Stats display (remains unchanged)
#     hp = st.session_state.state["hp"]
#     mana = st.session_state.state["mana"]
#     luck = st.session_state.state["luck"]

#     st.markdown(f"""
#     <div class="bottom-bar">
#         <div class="stat-container">
#             <strong>HP:</strong> {hp}/100
#         </div>
#         <div class="stat-container">
#             <strong>Mana:</strong> {mana}/100
#         </div>
#         <div class="stat-container">
#             <strong>Luck:</strong> {luck}/100
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
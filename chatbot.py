import os
import google.generativeai as genai
import streamlit as st
import pandas as pd
import joblib
from recommendation import preprocess_data

# Cache dataset and preprocessing components
def load_or_process_data(file_path):
    dataset_cache = 'dataset_cache.pkl'
    preprocess_cache = 'preprocessed_data.pkl'

    # Check if the dataset cache exists
    if os.path.exists(dataset_cache):
        print("Loading cached dataset...")
        df = joblib.load(dataset_cache)
    else:
        print("Loading and caching dataset...")
        df = pd.read_csv(file_path)
        joblib.dump(df, dataset_cache)

    # Check if the preprocessing cache exists
    if os.path.exists(preprocess_cache):
        print("Loading cached preprocessing components...")
        preprocessed_data = joblib.load(preprocess_cache)
    else:
        print("Preprocessing and caching components...")
        preprocessed_data = preprocess_data(df)
        joblib.dump(preprocessed_data, preprocess_cache)

    return (df,) + preprocessed_data

# Load or process the data
df, vectorizer, svd, scaler, knn = load_or_process_data('merged_data.csv')

API_KEY = 'AIzaSyBGLHeEYjGUS7x8AY_yulj5PZwGEcolHF4'


def initialize_chatbot():
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-pro')

    if 'chat' not in st.session_state:
        st.session_state['chat'] = model.start_chat(history=[])

    if 'messages' not in st.session_state:
        st.session_state.messages = []

def enhance_cooking_directions(directions):
    """Use the model to enhance the grammar and context of cooking directions."""
    prompt = f"Please rephrase and correct the following cooking directions to make them more clear and grammatically correct:\n\n{directions}"
    chat = st.session_state['chat']

    try:
        response = chat.send_message(prompt, stream=True)
        response.resolve()

        enhanced_directions = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                enhanced_directions += part.text

        return enhanced_directions
    except genai.GenerativeModel.BrokenResponseError:
        last_send, last_received = chat.rewind()
        return "I'm sorry, there was an issue processing the cooking directions. Please try again."


def get_gemini_response(question):
    # Access conversation history
    conversation_history = st.session_state.messages

    # prompt including history
    prompt = f"""
    You are an interactive recipe assistant.

**Conversation History:**\n

You will assist the user in preparing a recipe step by step. 

1. First, provide the list of ingredients needed for the recipe. Once the user confirms they have the ingredients, ask if they would like to start cooking.
   
2. After the user confirms, guide them through the preparation process by providing one step at a time. After each step, ask the user if they have completed the step and whether they would like to continue or need any further assistance.

3. This process continues until the recipe is fully prepared, ensuring the user is comfortable and supported throughout the cooking experience.

If the user asks a question that is not related to the recipe or if there are any spelling errors, ask the user for clarification or correct spelling before proceeding. Only start the preparation when the user confirms they are ready to begin.
"""

    for message in conversation_history:
        prompt += f"{message['role']}: {message['content']}\n"

    prompt += f""" 
    **User Question:** {question}
    """

    # Check if the question is about a specific recipe
    for index, row in df.iterrows():
        if row['recipe_name'].lower() in question.lower():
            # Enhance the cooking directions using the model
            enhanced_directions = enhance_cooking_directions(row['formatted_directions'])
            return f"Here are the improved cooking directions for {row['recipe_name']}:\n\n{enhanced_directions}"

    # If no matching recipe, use generative model for a general response
    chat = st.session_state['chat']

    try:
        response = chat.send_message(prompt, stream=True)
        response.resolve()

        full_response_text = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                full_response_text += part.text

        return full_response_text
    except genai.GenerativeModel.BrokenResponseError:
        last_send, last_received = chat.rewind()
        return "I'm sorry, there was an issue with your request. Please try again."

# Initialize the chatbot on app start
initialize_chatbot()
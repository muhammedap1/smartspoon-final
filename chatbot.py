import os
import google.generativeai as genai
import streamlit as st
import pandas as pd
import joblib
from recommendation import preprocess_data

# Load your dataset
df = pd.read_csv('merged_data.csv')

# Cache preprocessing components
def load_or_preprocess_data(data):
    # Check if the cache file exists
    cache_file = 'preprocessed_data.pkl'
    if os.path.exists(cache_file):
        vectorizer, svd, scaler, knn = joblib.load(cache_file)
    else:
        vectorizer, svd, scaler, knn = preprocess_data(data)
        joblib.dump((vectorizer, svd, scaler, knn), cache_file)

    return vectorizer, svd, scaler, knn

# Load or preprocess the data
vectorizer, svd, scaler, knn = load_or_preprocess_data(df)

def initialize_chatbot():
    genai.configure(api_key=os.getenv('google_api_key'))
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
    except genai.generative_models.BrokenResponseError:
        last_send, last_received = chat.rewind()
        return "I'm sorry, there was an issue processing the cooking directions. Please try again."

def get_gemini_response(question):
    # Access conversation history
    conversation_history = st.session_state.messages

    # Build the prompt including history
    prompt = f"""
    You are an interactive recipe assistant.

    **Conversation History:**\n

    You will assist the user in preparing a recipe step by step. First, provide the list of ingredients needed for the recipe. Once the user confirms they have the ingredients, ask if they would like to start cooking.

    After the user confirms, guide them through the preparation process by providing one step at a time. After each step, ask the user if they have completed the step and whether they would like to continue or need any further assistance.

    This process continues until the recipe is fully prepared, ensuring the user is comfortable and supported throughout the cooking experience.
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
    except genai.generative_models.BrokenResponseError:
        last_send, last_received = chat.rewind()
        return "I'm sorry, there was an issue with your request. Please try again."

# Initialize the chatbot on app start
initialize_chatbot()



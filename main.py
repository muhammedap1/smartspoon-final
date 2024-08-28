import streamlit as st
import pandas as pd
import random
from recommendation import preprocess_data, recommend_recipes, recipe_data
from chatbot import initialize_chatbot, get_gemini_response

# Initialize chatbot
initialize_chatbot()

# Preprocess data
vectorizer, svd, scaler, knn = preprocess_data(recipe_data)

# List of cooking tips and facts
cooking_tips = [
    "Always read the entire recipe before you start cooking.",
    "Mise en place: Prepare and measure all ingredients before you start cooking.",
    "A sharp knife is safer than a dull one as it requires less pressure to cut.",
    "Add a pinch of salt to bring out the flavor in sweet dishes.",
    "Let meat rest for a few minutes after cooking to retain its juices.",
    "Wooden cutting boards are gentler on knife blades than plastic ones.",
    "Room temperature eggs blend more easily in batters.",
    "Dried herbs are generally more potent than fresh ones.",
    "Acid (like lemon juice or vinegar) can balance out overly salty dishes.",
    "Roasting vegetables brings out their natural sweetness.",
]

def main():
    st.set_page_config(page_title="SmartSpoon🥄🍳👩‍🍳", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size: 70px !important;
        color: #FF69B4;
        font-weight: bold;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        padding: 20px 0;
        background: linear-gradient(45deg, #FFC0CB, #FF69B4);
        border-radius: 15px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #FF69B4;
        color: white;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#FF69B4, #FF1493);
    }
    .tip-box {
        background-color: #F0F8FF;
        border-left: 5px solid #FF69B4;
        padding: 10px;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 36px;
        color: #FF1493;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px #FFC0CB;
    }
    .subsection-header {
        font-size: 24px;
        color: #FF69B4;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<p class='big-font'>SmartSpoon🥄🍳👩‍🍳</p>", unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Recipe Finder", "Cook With ME", "Recipe of the Day", "Meal Planner","About"])

    
    if page == "Recipe Finder":
        recipe_finder()
    elif page == "Cook With ME":
        cook_with_me()
    elif page == "Recipe of the Day":
        recipe_of_the_day()
    elif page == "About":
        about_page()
    else:
        meal_planner()
        
def about_page():
    st.markdown("<p class='section-header'>About SmartSpoon</p>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to SmartSpoon, your all-in-one culinary companion! Our application is designed to make your cooking experience easier, more enjoyable, and tailored to your needs. Here's what you can do with SmartSpoon:

    ### 1. Recipe Finder
    Our Recipe Finder is a powerful tool that helps you discover the perfect dish based on your specific needs:
    - Input your nutritional requirements (calories, protein, carbs, fat, etc.)
    - Specify your desired preparation time
    - List ingredients you have or want to use
    The system will then find the best recipes that match your criteria, making meal planning a breeze!

    ### 2. Cook With ME - Your Culinary Assistant
    Get real-time guidance and answers to all your cooking questions with our conversational chatbot:
    - Ask anything about cooking techniques, ingredient substitutions, or recipe modifications
    - Get step-by-step guidance through the cooking process
    - Receive instant answers to your culinary queries, making you feel like you have a chef right by your side

    ### 3. Recipe of the Day
    Discover new and exciting dishes every day:
    - Get a randomly selected recipe each time you visit
    - View detailed nutritional information and ingredient list
    - Learn a new cooking tip with each recipe, expanding your culinary knowledge

    ### 4. Meal Planner
    Organize your weekly meals effortlessly:
    - Plan breakfast, lunch, and dinner for each day of the week
    - Add meals to specific days and meal types
    - Clear individual meals or the entire plan as needed
    - Export your meal plan as a CSV file for easy reference

    SmartSpoon is here to make your culinary journey more exciting and less stressful. Whether you're a beginner cook or a seasoned chef, our tools are designed to inspire and assist you in creating delicious meals tailored to your preferences and needs.

    Happy cooking with SmartSpoon! 🥄🍳👩‍🍳
    """)

def recipe_finder():
    st.markdown("<p class='section-header'>Recipe Finder</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px; color: #555; font-weight: bold; text-align: center;'>Enter the details below to get personalized recipe recommendations:</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        prep_time = st.slider('Preparation Time (minutes)', 0, 120, 30)
        calories = st.slider('Calories', 0, 1000, 500)
        fat = st.slider('Fat (g)', 0, 100, 20)
        carbohydrates = st.slider('Carbohydrates (g)', 0, 200, 50)

    with col2:
        protein = st.slider('Protein (g)', 0, 100, 30)
        cholesterol = st.slider('Cholesterol (mg)', 0, 300, 100)
        sodium = st.slider('Sodium (mg)', 0, 2000, 500)
        fiber = st.slider('Fiber (g)', 0, 50, 10)

    ingredients = st.text_area('Ingredients (comma-separated)', placeholder="e.g., chicken, rice, onion")

    if st.button('Find Recipes', key='find_recipes'):
        with st.spinner('Searching for the perfect recipes...'):
            input_features = [prep_time, calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
            recommendations = recommend_recipes(input_features, vectorizer, svd, scaler, knn, recipe_data)

        st.success("Here are your personalized recipe recommendations!")
        
        for idx, row in recommendations.iterrows():
            with st.expander(f"**{row['recipe_name']}**"):
                st.write(f"**Preparation Time:** {row['prep_time']} minutes")
                st.write(f"**Ingredients:** {row['ingredients_list']}")

def cook_with_me():
    st.markdown("<p class='section-header'>Cook With ME - Your Culinary Assistant</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px; font-weight: bold; text-align: center;'>Ask anything about cooking, recipes, or culinary tips!</p>", unsafe_allow_html=True)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What's your culinary question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in get_gemini_response(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def recipe_of_the_day():
    st.markdown("<p class='section-header'>Recipe of the Day</p>", unsafe_allow_html=True)
    
    # Randomly select a recipe
    random_recipe = recipe_data.sample(n=1).iloc[0]
    
    st.markdown(f"<p class='subsection-header'>{random_recipe['recipe_name']}</p>", unsafe_allow_html=True)
    
    # Display a random cooking tip
    tip = random.choice(cooking_tips)
    st.markdown(f"<div class='tip-box'><strong>Cooking Tip of the Day:</strong> {tip}</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Preparation Time:** {random_recipe['prep_time']} minutes")
        st.write(f"**Calories:** {random_recipe['calories']}")
        st.write(f"**Protein:** {random_recipe['protein']}g")
    
    with col2:
        st.write(f"**Carbohydrates:** {random_recipe['carbohydrates']}g")
        st.write(f"**Fat:** {random_recipe['fat']}g")
        st.write(f"**Sodium:** {random_recipe['sodium']}mg")
    
    st.markdown("<p class='subsection-header'>Ingredients</p>", unsafe_allow_html=True)
    st.write(random_recipe['ingredients_list'])

def meal_planner():
    st.markdown("<p class='section-header'>Meal Planner</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px; color: #555; font-weight: bold; text-align: center;'>Plan your meals for the week!</p>", unsafe_allow_html=True)

    # Initialize meal plan in session state if it doesn't exist
    if 'meal_plan' not in st.session_state:
        st.session_state.meal_plan = pd.DataFrame(
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=['Breakfast', 'Lunch', 'Dinner']
        )

    # Display the meal plan table
    st.markdown("<p class='subsection-header'>Weekly Meal Plan</p>", unsafe_allow_html=True)
    st.table(st.session_state.meal_plan.style.set_properties(**{'text-align': 'center'}))

    # Add meal to the plan
    st.markdown("<p class='subsection-header'>Add a Meal</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        day = st.selectbox("Select day", st.session_state.meal_plan.index)
    with col2:
        meal_type = st.selectbox("Select meal type", st.session_state.meal_plan.columns)
    with col3:
        meal = st.text_input("Enter meal")

    if st.button("Add to Plan"):
        if meal:  # Only add if a meal is entered
            st.session_state.meal_plan.loc[day, meal_type] = meal
            st.success(f"Added {meal} to {day}'s {meal_type}")
        else:
            st.warning("Please enter a meal before adding to the plan.")

    # Clear specific meal
    st.markdown("<p class='subsection-header'>Clear a Specific Meal</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        clear_day = st.selectbox("Select day to clear", st.session_state.meal_plan.index, key="clear_day")
    with col2:
        clear_meal_type = st.selectbox("Select meal type to clear", st.session_state.meal_plan.columns, key="clear_meal_type")

    if st.button("Clear Specific Meal"):
        st.session_state.meal_plan.loc[clear_day, clear_meal_type] = ""
        st.success(f"Cleared {clear_day}'s {clear_meal_type}")

    # Clear the entire meal plan
    if st.button("Clear Entire Meal Plan"):
        st.session_state.meal_plan = pd.DataFrame(
            index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            columns=['Breakfast', 'Lunch', 'Dinner']
        )
        st.success("Entire meal plan cleared!")

    # Export meal plan
    if st.button("Export Meal Plan"):
        csv = st.session_state.meal_plan.to_csv(index=True)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="meal_plan.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
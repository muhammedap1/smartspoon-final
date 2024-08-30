

---

# SmartSpoon 🍴

**SmartSpoon** is an AI-Powered Cooking Assistant that helps users discover and prepare recipes based on their ingredients, dietary preferences, and nutritional needs. The application provides step-by-step cooking instructions, preparation methods, and alternative suggestions, all through a conversational interface powered by a chatbot.

## Features

- **Recipe Recommendation**: Get personalized recipe suggestions based on your available ingredients and dietary preferences.
- **Recipe of the Day**: Receive a randomly selected recipe each time you use the app.
- **Meal Planner**: Create a meal plan that can be downloaded as a CSV file.
- **Interactive Chatbot**: The AI-powered chatbot interacts with users to provide cooking directions, answer queries, and offer alternatives.
- **About Page**: Learn more about the app and its functionalities.

## Challenges and Solutions

During the development of SmartSpoon, various machine learning models and techniques were tested to improve the application's performance:

- **Clustering Models**: Tried clustering models like DBSCAN and K-Means, but both produced too much noise and failed to classify correctly.
- **Chatbot Fine-Tuning**: Attempted to fine-tune models like GPT-2 and T5-small, but they repeated words and did not generate coherent responses.
- **RAG Architecture**: Experimented with the Retrieval-Augmented Generation (RAG) architecture, but it performed poorly in this context.
- **Final Approach**: The app retrieves data directly from the dataset. If a recipe is not found, the Gemini Pro model generates it. This approach improved the performance and user experience.

## Project Structure

- **`Statics/`**: Contains static files and assets used in the application.
- **`.env`**: Environment variables configuration file.
- **`README.md`**: This file, which provides an overview of the project.
- **`chatbot.py`**: Contains the implementation of the chatbot, including conversation logic and interaction with the user.
- **`colab.txt`**: Notes or instructions related to working with Google Colab (if applicable).
- **`dataset_cache.pkl`**: Pickle file storing the cached dataset for faster access.
- **`main.py`**: The main entry point for running the SmartSpoon application.
- **`merged_data.csv`**: CSV file containing the merged dataset used for recipe recommendations and cooking directions.
- **`preprocessed_data.pkl`**: Pickle file containing preprocessed data ready for model training or embedding.
- **`processed_data.pkl`**: Pickle file containing processed data used in the application.
- **`recommendation.py`**: Handles the logic for recipe recommendation, including the use of machine learning models.
- **`requirements.txt`**: List of Python dependencies required to run the application.

## Installation

To run SmartSpoon locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/muhammedap1/smartspoon-final.git
    cd smartspoon
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
   - Rename `.env.example` to `.env` and fill in the necessary environment variables.

5. **Run the application**:
    ```bash
    streamlit run main.py
    ```

## Usage

1. Launch the application and navigate to the `Recipe` page to explore recipe recommendations.
2. Use the `Cook With ME` page to interact with the chatbot and receive step-by-step cooking instructions.
3. Manage your ingredients list and customize recommendations based on your preferences.
4. Check out the `Recipe of the Day` for a new recipe suggestion every time you use the app.
5. Create and download a meal plan using the `Meal Planner` feature.
6. Visit the `About` page to learn more about the app.

## Technologies Used

- **Python**: Core programming language for application development.
- **Streamlit**: Framework for building the interactive web application.
- **FAISS**: Library for efficient similarity search and clustering of dense vectors.
- **Google Generative AI**: Used for enhancing chatbot responses and cooking directions.
- **Sentence Transformers**: Used for generating embeddings from text data.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, feel free to reach out:

- **Muhammed** - [muhammedap0752@gmail.com](mailto:your-email@example.com)

---

You can now copy and paste this into your README file. Don't forget to replace `"yourusername"` and `"your-email@example.com"` with your actual GitHub username and contact email.
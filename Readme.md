# Medical FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a medical FAQ CSV.

## Features
- Retrieval-Augmented Generation (RAG) with CSV knowledge base
- OpenAI GPT (`gpt-4o-mini`) for natural language answers
- Hugging Face embeddings (`all-MiniLM-L6-v2`) running locally (no key needed)
- Chroma vector database
- Streamlit UI with chat history

## Setup

1. Clone or unzip the project

2. Install dependencies:
pip install -r requirements.txt

3. .env.example → .env and add your OpenAI key:
OPENAI_API_KEY=sk-xxxxxx

4. Make sure MedicalQuestionAnswering.csv is in the project folder.

5. streamlit run app.py

6. Usage:
Ask a medical FAQ from the CSV → bot answers from the dataset.
Ask a follow-up → bot uses context to answer correctly.
Ask something outside the CSV → bot replies: “I don’t know based on the current knowledge base.”
Use the Clear chat button to reset the conversation.

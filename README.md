# Voice_Based_Cognitive_Decline_Pattern_Detection
This project detects early cognitive decline from voice clips using machine learning models. It involves preprocessing audio, converting speech to text, extracting features, and then applying unsupervised machine learning models. The project includes a FastAPI-based local deployment and a public deployment via Streamlit for easy interaction.

## Features
Audio preprocessing for noise reduction and segmentation.

Speech-to-text conversion for processing audio data.

Feature extraction from speech for cognitive analysis.

Detection of cognitive decline using unsupervised ML and NLP techniques.

FastAPI deployment for local usage.

Streamlit deployment for public access.

### Requirements
Make sure to install the following dependencies:
`pip install -r requirements.txt`
The requirements.txt file includes the necessary packages like fastapi, streamlit, pydantic, numpy, scikit-learn, and any additional dependencies required for your machine learning models.

## Setup
### Local Deployment (FastAPI)
To deploy the application locally using FastAPI, follow these steps:

Make sure your virtual environment is activated (if using one).

Run the FastAPI application with the following command:
`uvicorn main:app --reload`
This will start the FastAPI server at 'http://127.0.0.1:8000', where you can interact with the API locally.

You can test the endpoints by sending requests (e.g., using Postman or cURL) or navigate to 'http://127.0.0.1:8000/docs' for an interactive API documentation.

### Public Deployment (Streamlit)
To deploy the application using Streamlit publicly:

Install Streamlit if it’s not already installed:
`pip install streamlit`
Run the Streamlit app:
`streamlit run final_app.py`
Streamlit will launch the application in your browser, where users can upload voice clips for cognitive decline detection.

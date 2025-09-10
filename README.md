# 📊 NUSAScore - AI-Powered Credit Risk API

An API service that uses a machine learning model to provide real-time credit risk predictions. The model is trained on the classic German Credit dataset to automate and accelerate credit underwriting decisions.

This project was developed as an individual submission for the Astranauts 2025 competition.

## 🚀 Features

-   **Real-Time Prediction**: Exposes a `/predict` endpoint to score new loan applications instantly.
-   **High-Performance Model**: Utilizes a tuned **XGBoost Classifier** to achieve high accuracy in predicting credit risk.
-   **Actionable Business Logic**: Translates the raw probability score into a clear, suggested decision: **APPROVE**, **FLAG FOR REVIEW**, or **DECLINE**.
-   **Robust API**: Built with **FastAPI** and **Pydantic** for automatic data validation, documentation, and high performance.
-   **Reproducible Training**: Includes a dedicated script (`train.py`) to preprocess data and train the model from scratch.

## 🛠 Tech Stack

-   **Backend & API**: Python, FastAPI, Uvicorn
-   **Machine Learning**: scikit-learn, XGBoost, Pandas
-   **Data Validation**: Pydantic
-   **Serialization**: Joblib

## 📂 File Structure
├── german.doc                  # Documentation explaining the dataset features and codes.  
├── credit.data                 # The raw, space-delimited dataset.  
├── train.py                    # Script to preprocess data, train the XGBoost model, and save artifacts.  
├── main.py                     # The FastAPI application that loads the model and serves predictions.  
├── nusa_score_model.joblib     # Serialized, trained XGBoost model object.  
├── nusa_score_meta.joblib      # Metadata from training (e.g., column names) for consistent preprocessing.  
├── ui.py                       # Streamlit UI script  
└── README.md                   # You are here!  


## 🧪 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/teresakae/NUSAScore
    cd NUSAScore
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install fastapi uvicorn pandas scikit-learn xgboost joblib pydantic
    ```
3.  **Train the model:**
    Run the training script to generate the `.joblib` artifacts.
    ```bash
    python train.py
    ```
4.  **Run the API server:**
    ```bash
    uvicorn main:app --reload
    ```
5.  **Access the API:**
    Visit `http://127.0.0.1:8000/docs` in your browser to see the interactive API documentation and test the `/predict` endpoint.

## ✅ Future Improvements

-   **Develop a Front-End Dashboard**: Build a user interface with Streamlit for credit officers to easily input applicant data and view results.
-   **Containerize the Application**: Use Docker to containerize the API for consistent deployment and scalability.
-   **Cloud Deployment**: Deploy the API to a cloud service like Azure or AWS for public access and integration with other systems.
-   **Model Monitoring & Retraining**: Implement a system to monitor the model's performance over time and establish a pipeline for periodic retraining on new data.

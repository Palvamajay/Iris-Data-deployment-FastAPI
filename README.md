# Iris-Data-deployment-FastAPI
# 🌸 Iris Flower Classifier – FastAPI Deployment
📌 Project Overview
This project demonstrates the complete workflow of building and deploying a machine learning model using the Iris dataset. The goal is to classify iris flower species (Setosa, Versicolor, and Virginica) based on sepal and petal measurements. The trained model is deployed as a FastAPI web service, enabling real-time predictions via HTTP requests.

# 🎯 Objectives
Analyze and understand the Iris dataset through Exploratory Data Analysis (EDA).

Train a Logistic Regression model to classify iris species.

Deploy the trained model as a REST API using FastAPI.

Enable easy access to predictions through an API endpoint.

# 📊 Exploratory Data Analysis
Key insights from EDA:

Petal length and petal width are highly correlated (0.96) and are strong indicators for species classification.

Sepal width shows less correlation with other features.

Box plots, violin plots, and scatter plots reveal clear separation between species.

# 🤖 Model Building
Algorithm: Logistic Regression (Multi-class classification)

Reason for choice:

Simple and interpretable

Performs well with small, clean datasets

Produces probability estimates for predictions

# ⚡ FastAPI Deployment
FastAPI is a modern, high-performance Python framework for building APIs. In this project:

Trained model saved using pickle.

FastAPI app created to handle prediction requests.

POST endpoint /predict accepts input JSON with flower measurements.

Returns the predicted species name.

API tested locally using Uvicorn and Swagger UI

# 🚀 Challenges Faced
Structuring the project for model + API integration.

Understanding Pydantic for data validation.

Debugging API routes and imports.

Learning to use Swagger UI for interactive testing.

# 🏷 Tech Stack
Python

Scikit-learn

FastAPI

Pydantic

Uvicorn

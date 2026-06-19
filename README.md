# Titanic Survival Predictor

A machine learning web application that predicts whether a Titanic passenger would survive based on historical passenger data.

## Features

* Survival prediction using Random Forest
* Probability estimation
* Historical passenger comparison
* Interactive web interface built with FastAPI
* Titanic-inspired archive design

## Technologies

* Python
* Pandas
* Scikit-learn
* FastAPI
* HTML/CSS

## Dataset

The project uses the Titanic dataset from Kaggle.

## Machine Learning Model

Algorithm:

* Random Forest Classifier

Features:

* Passenger Class
* Gender
* Age
* Ticket Fare
* Number of Siblings/Spouse
* Number of Parents/Children

Model Accuracy:

* 80.4%

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

http://127.0.0.1:8000

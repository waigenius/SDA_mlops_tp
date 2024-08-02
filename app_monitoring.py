from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
##
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments
##
from dotenv import load_dotenv
load_dotenv()
import datetime

ARIZE_SPACE_KEY=os.getenv("SPACE_KEY")
ARIZE_API_KEY = os.getenv("API_KEY")

# Initialize Arize client with your space key and api key
arize_client = Client(space_key=ARIZE_SPACE_KEY, api_key=ARIZE_API_KEY)

# Define the schema for your data
schema = Schema(
    prediction_id_column_name="prediction_id",
    timestamp_column_name="timestamp",
    feature_column_names=["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"],
    prediction_label_column_name="prediction_label",
    actual_label_column_name="actual_label"
)


app = Flask(__name__)
model = pickle.load(open("catboost_model-2.pkl", "rb"))


def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Age = int(request.form["Age"])
        RestingBP = int(request.form["RestingBP"])
        Cholesterol = int(request.form["Cholesterol"])
        Oldpeak = float(request.form["Oldpeak"])
        FastingBS = int(request.form["FastingBS"])
        MaxHR = int(request.form["MaxHR"])

        # Assume you have actual labels available for evaluation
        actual_label = int(request.form.get("actual_label for evaluation", 1))  # Default to -1 if not provided
        #

        prediction = model.predict(
            [[Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak]]
        )
        # Log the prediction to Arize
        timestamp = pd.Timestamp.now()

        # Log the prediction to Arize
        data = {
            "prediction_id": [str(timestamp.timestamp())],  # Unique ID for each prediction
            "timestamp": [timestamp],
            "Age": [Age],
            "RestingBP": [RestingBP],
            "Cholesterol": [Cholesterol],
            "FastingBS": [FastingBS],
            "MaxHR": [MaxHR],
            "Oldpeak": [Oldpeak],
            "prediction_label": [int(prediction[0])],
            "actual_label": [actual_label]             
        }
        dataframe = pd.DataFrame(data)

        # Debug prints
        print("DataFrame:")
        print(dataframe)
        print("Schema:")
        print(schema)
        
        try: 
            response = arize_client.log(
                dataframe = dataframe,
                model_id="your_model_id",
                model_version="v1",
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=Environments.PRODUCTION,
                #features=features,
                #prediction_label = [int(prediction[0])],
                schema=schema
            )

            if response.status_code != 200:
                print(f"Failed to log data to Arize: {response.text}")
            else:
                print("Successfully logged data to Arize")
        except Exception as e:
            print(f"An error occured: {e}")
        
        if prediction[0] == 1:
            return render_template(
                "index.html",
                prediction_text="Kindly make an appointment with the doctor!",
            )

        else:
            return render_template(
                "index.html", prediction_text="You are well. No worries :)"
            )

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

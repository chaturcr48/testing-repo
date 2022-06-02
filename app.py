# from pyexpat import model
# from textwrap import fill
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app=Flask(__name__)

lr = joblib.load("model.pkl")
print("Model loaded")
model_columns = joblib.load("model_columns.pkl")
print("Model columns loaded")


@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            """
            input
            [
                {"Age": 33, "Sex": "Female", "Embarked": "S"},
                {"Age": 63, "Sex": "Female", "Embarked": "C"}
            ]

            """

            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print("Train the model first")
        return ("No model here to use")


if __name__ == '__main__':
	app.run(debug=True)
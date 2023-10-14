import pickle
import numpy as np
from flask import Flask, request, jsonify


def predict_single(customer, dv, model):
    X = dv.transform([customer])  ## apply the one-hot encoding feature to the customer data 
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


app = Flask('app') # give an identity to your web service

with open('./dv.bin', 'rb') as f:
    dv = pickle.load(f)

with open('./model1.bin', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
def predict():
    customer = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.

    prediction = predict_single(customer, dv, model)
    res = prediction >= 0.5

    result = {
        'res_probability': float(prediction), ## we need to cast numpy float type to python native float type
        'res': bool(res),  ## same as the line above, casting the value using bool method
    }

    return jsonify(result)  ## send back the data in json format to the user


if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696

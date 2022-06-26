# Melroy pereira


# Importing libraries
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from flask import Flask,render_template,request,jsonify


# app
app = Flask(__name__)

# model
model = pickle.load(open('model.pkl', 'rb'))

#scaler Y
scaling = pickle.load(open("scalerY.pkl", 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    array_features = [np.array(features)]
    prediction = model.predict(array_features)
    prediction = scaling.inverse_transform(prediction.reshape(-1,1))
    output = round(prediction[0][0],2)
    return render_template('index.html', prediction_text='The predicted demand in rented bikes is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)


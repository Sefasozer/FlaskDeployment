import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

if __name__ == '__main__': 
    os.environ.setdefault('FLASK_ENV', 'development')
    
    
    

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.plk", "rb"))

@flask_app.route("/")
def index():
    return render_template("home.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("home.html", result_of_prediction = "price {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)
    


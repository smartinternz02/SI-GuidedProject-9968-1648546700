import re
import numpy as np
import os
from flask import Flask, app, request, render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for
# Loading the model

model1 = load_model("Model1.h5")
model2 = load_model("LevelModel.h5")

app = Flask(__name__)

# default home page or route


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def home():
    return render_template("index.html")


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/result', methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        # getting the current path i.e where app.py is present
        basepath = os.path.dirname(__file__)
        #print("current path",basepath)
        # from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        filepath = os.path.join(basepath, 'uploads', f.filename)
        #print("upload folder is",filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)  # img to array
        x = np.expand_dims(x, axis=0)  # used for adding one more dimension
        # print(x)
        img_data = preprocess_input(x)
        print(model1.predict(img_data), model2.predict(img_data))
        prediction1 = np.argmax(model1.predict(img_data))
        prediction2 = np.argmax(model2.predict(img_data))
        print(prediction1, prediction2)
        # prediction=model.predict(x)#instead of predict_classes(x) we can use predict(X) ---->predict_classes(x) gave error
        #print("prediction is ",prediction)
        index1 = ['front', 'rear', 'side']
        index2 = ['minor', 'moderate', 'severe']
        #result = str(index[output[0]])
        result1 = index1[prediction1]
        result2 = index2[prediction2]
        print(result1, result2)
        if(result1 == "front" and result2 == "minor"):
            number = "3000 - 5000 INR"

        elif(result1 == "front" and result2 == "moderate"):
            number = "6000 - 8000 INR"

        elif(result1 == "front" and result2 == "severe"):
            number = "9000 - 11000 INR"

        elif(result1 == "rear" and result2 == "minor"):
            number = "4000 - 6000 INR"

        elif(result1 == "rear" and result2 == "moderate"):
            number = "7000 - 10000 INR"

        elif(result1 == "rear" and result2 == "severe"):
            number = "11000 - 13000 INR"

        elif(result1 == "side" and result2 == "minor"):
            number = "6000 - 8000 INR"

        elif(result1 == "side" and result2 == "moderate"):
            number = "9000 - 11000 INR"

        elif(result1 == "side" and result2 == "severe"):
            number = "12000 - 15000 INR"

        else:
            number = "15000 - 50000 INR"

        return render_template('prediction.html', prediction=number)


""" Running our application """
if __name__ == "__main__":
    app.run(port=8081)

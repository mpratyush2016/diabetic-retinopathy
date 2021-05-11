from math import e
from flask import Flask, render_template, render_template_string , request , url_for , send_from_directory , send_file , flash, redirect, jsonify, make_response
import numpy as np
import pandas as pd
import os
from flask import Flask, flash, request, redirect, url_for
from pyrebase.pyrebase import Database
from werkzeug.utils import secure_filename
# from pathlib import Path
# import os.path
# import matplotlib.pyplot as plt
# from IPython.display import Image, display, Markdown
# import matplotlib.cm as cm
# from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
import tensorflow as tf
# from time import perf_counter
# import seaborn as sns
from keras.models import load_model
print(tf.version.VERSION)
import pyrebase

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['SECRET_KEY'] = 'somethingsecret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

config = {
    "apiKey": "AIzaSyBDx_WUowp0AqQ7tPWL4gzDIrIQsI_xAXM",
    "authDomain": "diabetic-retinopathy-d2355.firebaseapp.com",
    "databaseURL": "gs://diabetic-retinopathy-d2355.appspot.com",
    "projectId": "diabetic-retinopathy-d2355",
    "storageBucket": "diabetic-retinopathy-d2355.appspot.com",
    "messagingSenderId": "200185786361",
    "appId": "1:200185786361:web:6982781e4fa26c0cfd5728",
    "measurementId": "G-5BG336DT03"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

def create_df(filepath):
    list1 = []
    list1.append(filepath)
    list2 = ['']
    df = pd.DataFrame(list(zip(list1, list2)), columns = ['FilePath', 'Label'])
    return df

labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3: 'Proliferate_DR', 4: 'Severe'}
image_df = pd.read_csv('image.csv')

train_df, test_df = train_test_split(image_df, train_size = 0.9, shuffle = True, random_state = 1)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

def get_prediction(image_path):
    image_path = image_path
    test_df = create_df(image_path)
    print(test_df)
    test_images = test_generator.flow_from_dataframe(
        dataframe = test_df,
        x_col = 'FilePath',
        y_col = 'Label',
        target_size = (224, 224),
        color_mode = 'rgb',
        class_mode = 'categorical',
        batch_size = 32,
        shuffle = False,
    )

    model = tf.keras.models.load_model('super_new_model')
    pred = model.predict(test_images)
    confidence = max(pred[0])*100
    pred = np.argmax(pred, axis = 1)
    pred = [labels[k] for k in pred]
    pred = pred[0]

    json = []

    json.append(
        {
            "category":pred,
            "confidence":confidence,
        }
    )
    return json


@app.route('/')
def home():
    storage.child('eye.png').download('uploads/','eye.png')
    json_data = get_prediction('eye.png')
    if os.path.exists('eye.png'):
        os.remove('eye.png')
    return render_template('sample.html', print = json_data)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/desktop', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = 'uploads/'+filename
            json = get_prediction(filepath)
            print(json)
            print("WORKING!")
            if os.path.exists(filepath):
                os.remove(filepath)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug = True)
import os
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from tensorflow import keras
from flask_cors import CORS
from tensorflow.keras.models import load_model
from bidict import bidict
from random import choice
from werkzeug.utils import secure_filename
import cv2
import requests

app = Flask(__name__)
app.secret_key = 'alphalearner'


CORS(app)



UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



ENCODER = bidict({
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18,
    'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
    'Y': 25, 'Z': 26,
    'a': 27, 'b': 28, 'c': 29, 'd': 30, 'e': 31, 'f': 32,
    'g': 33, 'h': 34, 'i': 35, 'j': 36, 'k': 37, 'l': 38,
    'm': 39, 'n': 40, 'o': 41, 'p': 42, 'q': 43, 'r': 44,
    's': 45, 't': 46, 'u': 47, 'v': 48, 'w': 49, 'x': 50,
    'y': 51, 'z': 52,
    '0': 53, '1': 54, '2': 55, '3': 56, '4': 57,
    '5': 58, '6': 59, '7': 60, '8': 61, '9': 62,
    'AM':63,'GLA':64,'One':65,'Two':66
})

LETTER_MODEL_URL = "https://s3.ap-south-1.amazonaws.com/letter.h5/letter.h5"
LETTER_MODEL_PATH = "letter.h5"

def download_model():
    if not os.path.exists(LETTER_MODEL_PATH):
        print("Downloading letter.h5 from S3...")
        try:
            r = requests.get(LETTER_MODEL_URL, stream=True)
            r.raise_for_status()
            with open(LETTER_MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download letter.h5: {e}")

download_model()




@app.route('/')
def home():
    return render_template('home.html')
# ---------------------------------------------------------- Add Data ------------------------------------------------------------------------------
@app.route('/add-data', methods=['GET'])
def add_data():
   
    message = session.pop('message', '')
    labels = np.load('data/labels.npy')
    label_count = {label: 0 for label in ENCODER.keys()}
    for label in labels:
        label_count[label] += 1

    
    sorted_labels = sorted(label_count.items(), key=lambda x: x[1])
    suggested_letter = sorted_labels[0][0]

 
   

    return render_template("add_data.html", letter=suggested_letter, message=message)


@app.route('/add-data', methods=['POST'])
def add_data_post():
    label = request.form.get('letter')
    labels = np.load('data/labels.npy')
    labels = np.append(labels, label)
    np.save('data/labels.npy', labels)
    pixels = request.form.get('pixels')
    pixel_array = np.array(pixels.split(',')).astype(float).reshape(1, 50, 50)
    images = np.load('data/images.npy')
    images = np.vstack([images, pixel_array])
    np.save('data/images.npy', images)
    session['message'] = f'"{label}" added to the training dataset'

    return redirect(url_for('add_data'))



# ----------------------------------------------------------------- Practice Data ----------------------------------------------------------------

@app.route('/practice', methods=['GET'])
def practice():
    return render_template("practice.html", letter=None)


@app.route('/practice', methods=['POST'])
def handle_practice_input():
    pixel_data = request.form.get('pixels', '')
    pixel_array = np.array(pixel_data.split(','), dtype=float).reshape(1, 50, 50, 1)

    model = keras.models.load_model('letter.h5')
    prediction = model.predict(pixel_array)
    predicted_index = np.argmax(prediction, axis=-1)[0]
    predicted_letter = ENCODER.inverse[predicted_index]

    return render_template("practice.html", letter=predicted_letter)




@app.route('/quiz', methods=['GET'])
def quiz():
    score = 0
    question_no = 1
    letter = choice(list(ENCODER.keys()))
    return render_template("quiz.html", letter=letter, score=score, question_no=question_no)

@app.route('/quiz', methods=['POST'])
def quiz_post():
    letter = request.form['letter']
    score = int(request.form['score'])
    question_no = int(request.form['question_no'])

    pixels = request.form['pixels']
    pixels = pixels.split(',')
    img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

    model = keras.models.load_model('letter.h5')
    pred_letter = np.argmax(model.predict(img), axis=-1)
    pred_letter = ENCODER.inverse[pred_letter[0]]

    if pred_letter == letter:
        score += 1

    question_no += 1
    if question_no > 5:
        return render_template("quiz_result.html", score=score)

    next_letter = choice(list(ENCODER.keys()))
    return render_template("quiz.html", letter=next_letter, score=score, question_no=question_no)



@app.route('/word')
def word():
    return render_template('word.html')

def predict_captcha(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = cv2.resize(img, (200, 50))  
        img = img / 255.0 
    else:
        return "Image not valid"

    img = img[np.newaxis, :, :, np.newaxis]

    
    res = np.array(model1.predict(img))
    result = np.reshape(res, (5, 36))  
    k_ind = [np.argmax(i) for i in result]
    return ''.join([character[k] for k in k_ind])


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    prediction = predict_captcha(filepath)

    return jsonify({'prediction': prediction})
       

    

if __name__ == '__main__':
    app.run(debug=True)


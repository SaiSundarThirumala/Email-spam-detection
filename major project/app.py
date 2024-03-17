from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

app = Flask(__name__)

# Load the pre-trained model
model = load_model('spam_detection_model.h5')  # Replace 'your_model.h5' with the actual path to your trained model file

# Load the Tokenizer
tok = Tokenizer(num_words=1000)  # Assuming you want to use the same Tokenizer configuration

# Define the maximum length for padding sequences
max_len = 150

# Function to preprocess input text
def preprocess_text(text):
    text_sequences = tok.texts_to_sequences([text])
    text_sequences_matrix = sequence.pad_sequences(text_sequences, maxlen=max_len)
    return text_sequences_matrix

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML template for the input form

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_content = request.form['message']
        processed_input = preprocess_text(text_content)
        prediction = model.predict(processed_input)
        
        if prediction > 0.5:
            result = "This is a Spam mail"
        else:
            result = "This is not a Spam mail"

        return render_template('index.html', result=result, message=text_content)

if __name__ == '__main__':
    app.run(debug=True)

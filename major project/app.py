from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

app = Flask(__name__)

# Load the data and preprocess
data = pd.read_csv("C:\\Users\\SUNDAR\\Desktop\\codes\\major project\\spam.csv", encoding='latin1')
data['lab'] = data["Category"].replace({"ham": 0, "spam": 1})
X = data.Message
Y = data.Category
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)

X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.20)

# Tokenization and model setup
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences_matrix = sequence.pad_sequences(tok.texts_to_sequences(X_train), maxlen=max_len)

def rnn():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

model = rnn()
model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10, validation_split=0.2,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        textx = tok.texts_to_sequences([message])
        textx = sequence.pad_sequences(textx, maxlen=max_len)
        pred = model.predict(textx)
        result = "This is a Spam mail" if pred > 0.5 else "This is not a Spam mail"
        return render_template('index.html', result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)

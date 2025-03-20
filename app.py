from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model('sentiment_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    seq = tokenizer.texts_to_sequences([data])
    padded = pad_sequences(seq, maxlen=100)  # Adjust maxlen based on training
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return jsonify({'sentiment': sentiment, 'score': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

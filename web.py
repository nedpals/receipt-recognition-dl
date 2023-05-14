import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras
import cv2

image_size = 224
providers = ['gcash', 'paymaya']

title = 'Mobile Wallet Receipt Classifier'
description = 'Recognize if receipt is gcash or paymaya'

models = [
    keras.models.load_model('models/receipt_identify_stable1.h5'),
    keras.models.load_model('models/receipt_identify_v2.h5'),
    keras.models.load_model('models/receipt_identify_v3.h5'),
    keras.models.load_model('models/receipt_identify_v4.h5')
]
    
model_keys = ['stable1', 'stable2', 'v3', 'v4']

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route("/")
def index_page():
    return render_template('index.html', title=title, description=description)

@app.post("/prediction")
def predict_inputs():
    # model_used = request.form.get('classifier')
    input_file = request.files.get('input_file')
    input_data = cv2.imdecode(np.frombuffer(input_file.read(), np.uint8), cv2.IMREAD_COLOR)
    input_data = cv2.resize(input_data, (image_size, image_size))
    final_input = np.expand_dims(input_data, axis=0)
    raw_prediction = models[0].predict(final_input)
    predictions = {}

    for idx, name in enumerate(model_keys):
        raw_prediction = models[idx].predict(final_input)
        print(name, raw_prediction)
        predictions[name] = providers[int(raw_prediction[0][0])]

    return {
        'predictions': predictions
    }

if __name__ == "__main__":
    app.run()
